#include "renderer.h"
#include <algorithm>
#include <algorithm> // std::fill
#include <cmath>
#include <functional>
#include <glm/common.hpp>
#include <glm/gtx/component_wise.hpp>
#include <iostream>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tuple>

namespace render {

// The renderer is passed a pointer to the volume, gradinet volume, camera and an initial renderConfig.
// The camera being pointed to may change each frame (when the user interacts). When the renderConfig
// changes the setConfig function is called with the updated render config. This gives the Renderer an
// opportunity to resize the framebuffer.
Renderer::Renderer(
    const volume::Volume* pVolume,
    const volume::GradientVolume* pGradientVolume,
    const render::RayTraceCamera* pCamera,
    const RenderConfig& initialConfig)
    : m_pVolume(pVolume)
    , m_pGradientVolume(pGradientVolume)
    , m_pCamera(pCamera)
    , m_config(initialConfig)
{
    resizeImage(initialConfig.renderResolution);
}

// Set a new render config if the user changed the settings.
void Renderer::setConfig(const RenderConfig& config)
{
    if (config.renderResolution != m_config.renderResolution)
        resizeImage(config.renderResolution);

    m_config = config;
}

// Resize the framebuffer and fill it with black pixels.
void Renderer::resizeImage(const glm::ivec2& resolution)
{
    m_frameBuffer.resize(size_t(resolution.x) * size_t(resolution.y), glm::vec4(0.0f));
}

// Clear the framebuffer by setting all pixels to black.
void Renderer::resetImage()
{
    std::fill(std::begin(m_frameBuffer), std::end(m_frameBuffer), glm::vec4(0.0f));
}

// Return a VIEW into the framebuffer. This view is merely a reference to the m_frameBuffer member variable.
// This does NOT make a copy of the framebuffer.
gsl::span<const glm::vec4> Renderer::frameBuffer() const
{
    return m_frameBuffer;
}

// Main render function. It computes an image according to the current renderMode.
// Multithreading is enabled in Release/RelWithDebInfo modes. In Debug mode multithreading is disabled to make debugging easier.
void Renderer::render()
{
    resetImage();

    static constexpr float sampleStep = 1.0f;
    const glm::vec3 planeNormal = -glm::normalize(m_pCamera->forward());
    const glm::vec3 volumeCenter = glm::vec3(m_pVolume->dims()) / 2.0f;
    const Bounds bounds { glm::vec3(0.0f), glm::vec3(m_pVolume->dims() - glm::ivec3(1)) };

    // 0 = sequential (single-core), 1 = TBB (multi-core)
#ifdef NDEBUG
    // If NOT in debug mode then enable parallelism using the TBB library (Intel Threaded Building Blocks).
#define PARALLELISM 1
#else
    // Disable multi threading in debug mode.
#define PARALLELISM 0
#endif

#if PARALLELISM == 0
    // Regular (single threaded) for loops.
    for (int x = 0; x < m_config.renderResolution.x; x++) {
        for (int y = 0; y < m_config.renderResolution.y; y++) {
#else
    // Parallel for loop (in 2 dimensions) that subdivides the screen into tiles.
    const tbb::blocked_range2d<int> screenRange { 0, m_config.renderResolution.y, 0, m_config.renderResolution.x };
        tbb::parallel_for(screenRange, [&](tbb::blocked_range2d<int> localRange) {
        // Loop over the pixels in a tile. This function is called on multiple threads at the same time.
        for (int y = std::begin(localRange.rows()); y != std::end(localRange.rows()); y++) {
            for (int x = std::begin(localRange.cols()); x != std::end(localRange.cols()); x++) {
#endif
            // Compute a ray for the current pixel.
            const glm::vec2 pixelPos = glm::vec2(x, y) / glm::vec2(m_config.renderResolution);
            Ray ray = m_pCamera->generateRay(pixelPos * 2.0f - 1.0f);

            // Compute where the ray enters and exists the volume.
            // If the ray misses the volume then we continue to the next pixel.
            if (!instersectRayVolumeBounds(ray, bounds))
                continue;

            // Get a color for the current pixel according to the current render mode.
            glm::vec4 color {};
            switch (m_config.renderMode) {
            case RenderMode::RenderSlicer: {
                color = traceRaySlice(ray, volumeCenter, planeNormal);
                break;
            }
            case RenderMode::RenderMIP: {
                color = traceRayMIP(ray, sampleStep);
                break;
            }
            case RenderMode::RenderMIDA: {
                color = traceRayMIDA(ray, sampleStep);
                break;
            }
            case RenderMode::RenderComposite: {
                color = traceRayComposite(ray, sampleStep);
                break;
            }
            case RenderMode::RenderIso: {
                color = traceRayISO(ray, sampleStep);
                break;
            }
            case RenderMode::RenderTF2D: {
                color = traceRayTF2D(ray, sampleStep);
                break;
            }
            };
            // Write the resulting color to the screen.
            fillColor(x, y, color);

#if PARALLELISM == 1
        }
    }
});
#else
            }
        }
#endif
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// This function generates a view alongside a plane perpendicular to the camera through the center of the volume
//  using the slicing technique.
glm::vec4 Renderer::traceRaySlice(const Ray& ray, const glm::vec3& volumeCenter, const glm::vec3& planeNormal) const
{
    const float t = glm::dot(volumeCenter - ray.origin, planeNormal) / glm::dot(ray.direction, planeNormal);
    const glm::vec3 samplePos = ray.origin + ray.direction * t;
    const float val = m_pVolume->getSampleInterpolate(samplePos);
    return glm::vec4(glm::vec3(std::max(val / m_pVolume->maximum(), 0.0f)), 1.f);
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// Function that implements maximum-intensity-projection (MIP) raycasting.
// It returns the color assigned to a ray/pixel given it's origin, direction and the distances
// at which it enters/exits the volume (ray.tmin & ray.tmax respectively).
// The ray must be sampled with a distance defined by the sampleStep
glm::vec4 Renderer::traceRayMIP(const Ray& ray, float sampleStep) const
{
    float maxVal = 0.0f;

    // Incrementing samplePos directly instead of recomputing it each frame gives a measureable speed-up.
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;
    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        const float val = m_pVolume->getSampleInterpolate(samplePos);
        maxVal = std::max(val, maxVal);
    }

    // Normalize the result to a range of [0 to mpVolume->maximum()].
    return glm::vec4(glm::vec3(maxVal) / m_pVolume->maximum(), 1.0f);
}

// Function that implements maximum-intensity-difference-accumulation (MIDA) raycasting.
// It returns the color assigned to a ray/pixel given it's origin, direction and the distances
// at which it enters/exits the volume (ray.tmin & ray.tmax respectively).
// The ray must be sampled with a distance defined by the sampleStep
glm::vec4 Renderer::traceRayMIDA(const Ray& ray, float sampleStep) const
{
    float maxVal = 0.0f, opacity = 0.0f, delta = 0.0f, beta = 0.0f;
    glm::vec3 color = { 0.0f, 0.0f, 0.0f };

    // Incrementing samplePos directly instead of recomputing it each frame gives a measureable speed-up.
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;
    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        const float val = m_pVolume->getSampleInterpolate(samplePos) / m_pVolume->maximum();
        //if new max value is found
        if (val > maxVal) {
            //color from transfer function
            glm::vec4 transferFunction = getTFValue(m_pVolume->getSampleInterpolate(samplePos));
            glm::vec3 colorAtSample = { transferFunction[0], transferFunction[1], transferFunction[2] };
            float opacityAtSample = transferFunction[3];
            //the bigger the increase in the max value, the more visible it is
            delta = val - maxVal;
            maxVal = val;
            
            /* Transition between DVR and MIDA
            if (m_config.gamma <= 0) {
                beta = 1 - (delta * (1 + m_config.gamma));             
                color = beta * color + (1 - (beta * opacity)) * colorAtSample * opacityAtSample;
                opacity = beta * opacity + (1 - (beta * opacity)) * opacityAtSample;
            } else {*/

            //Transition between MIDA and MIP
            beta = 1 - delta;
            color = beta * color + (1 - (beta * opacity)) * colorAtSample * opacityAtSample;
            opacity = beta * opacity + (1 - (beta * opacity)) * opacityAtSample;
            //}   
        }
    }

    // If in the region of transition between MIDA and MIP, final accumulated color is interpolated with the color of the
    //maximum value after ray is transversed, with gamma as interpolation weight.
    if (m_config.gamma > 0) {
        glm::vec4 interpolatedColor = interpolateColor(glm::vec4 { color, opacity }, glm::vec4(glm::vec3(maxVal), 1.0f), m_config.gamma);
        return interpolatedColor;
    } else {
        return glm::vec4(color, opacity);
    }

}

// This functions interpolates two opacity weighted colors according to a factor
// Returns the new color + opacity.
glm::vec4 Renderer::interpolateColor(glm::vec4 color1, glm::vec4 color2, float factor) 
{
    glm::vec4 newColor = {};

    newColor[0] = (color1[0] * (color1[3]) * (1 - factor) + color2[0] * (color2[3]) * factor) / ((color1[3] * (1 - factor) + color2[3] * factor));
    newColor[1] = (color1[1] * (color1[3]) * (1 - factor) + color2[1] * (color2[3]) * factor) / ((color1[3] * (1 - factor) + color2[3] * factor));
    newColor[2] = (color1[2] * (color1[3]) * (1 - factor) + color2[2] * (color2[3]) * factor) / ((color1[3] * (1 - factor) + color2[3] * factor));
    newColor[3] = color1[3] * (1 - factor) + color2[3] * factor;

    return newColor;
}

// ======= TODO: IMPLEMENT ========
// This function should find the position where the ray intersects with the volume's isosurface.
// If volume shading is DISABLED then simply return the isoColor.
// If volume shading is ENABLED then return the phong-shaded color at that location using the local gradient (from m_pGradientVolume).
//   Use the camera position (m_pCamera->position()) as the light position.
// Use the bisectionAccuracy function (to be implemented) to get a more precise isosurface location between two steps.
glm::vec4 Renderer::traceRayISO(const Ray& ray, float sampleStep) const
{
    // The color used to color the isosurface
    static constexpr glm::vec3 isoColor { 0.8f, 0.8f, 0.2f };

    // Get the isovalue from the GUI
    float isoVal = m_config.isoValue;

    // Incrementing samplePos directly instead of recomputing it each frame gives a measureable speed-up.
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;

    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {

        // Interpolate the intensity value at the current position
        const float val = m_pVolume->getSampleInterpolate(samplePos);
        // If the ray hits the isovalue, return the isocolor
        if (val >= isoVal) {
            if (m_config.volumeShading) {
                //Regular volume Shading
                glm::vec3 lighDirection = samplePos - m_pCamera->position();
                float tBissection = bisectionAccuracy(ray, t - sampleStep, t, isoVal);
                samplePos = ray.origin + tBissection * ray.direction;
                glm::vec3 shade = computePhongShading(isoColor, m_pGradientVolume->getGradientInterpolate(samplePos), lighDirection, lighDirection);
                return glm::vec4(shade, 1.0f);
            } else if (m_config.Warm2CoolShading) {
                //Warm to Cool Shading
                //position of light is hardcoded to be on top of object and below
                glm::vec3 lighDirection = samplePos - glm::vec3 { 128, 64, 128 }; // carp
                //glm::vec3 lighDirection = samplePos - glm::vec3 { 128, 0, 90 }; // pork
                glm::vec3 viewDirection = samplePos - m_pCamera->position();
                float tBissection = bisectionAccuracy(ray, t - sampleStep, t, isoVal);
                samplePos = ray.origin + tBissection * ray.direction;
                glm::vec3 shade = computePhongShadingWarm2Cool(isoColor, m_pGradientVolume->getGradientInterpolate(samplePos), lighDirection, viewDirection, m_config.alphaValue, m_config.betaValue, m_config.warmColor, m_config.coolColor);
                return glm::vec4(shade, 1.0f);
            } 
            else {
                return glm::vec4(isoColor, 1.0f);
            }
        }
    }
    // If the ray never encounters the isovalue, return black
    return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
}


// ======= TODO: IMPLEMENT ========
// Given that the iso value lies somewhere between t0 and t1, find a t for which the value
// closely matches the iso value (less than 0.01 difference). Add a limit to the number of
// iterations such that it does not get stuck in degerate cases.
float Renderer::bisectionAccuracy(const Ray& ray, float t0, float t1, float isoValue) const
{
    int i = 0;
    float initialT1 = t1;

    while (i < 1000) {
        float tMid = (t1 + t0) / 2;
        float val = m_pVolume->getSampleInterpolate(ray.origin + tMid * ray.direction);

        if (abs(isoValue - val) < 0.01) {
            return tMid;
        }
        else if(val > isoValue){
            t1 = tMid;
        }
        else {
            t0 = tMid;
        }
        i++;
    }
 
    return initialT1;
}

glm::vec3 Renderer::computePhongShading(const glm::vec3& color, const volume::GradientVoxel& gradient, const glm::vec3& L, const glm::vec3& V)
{
    glm::vec3 lighColor = glm::vec3(1.0f, 1.0f, 1.0f);
    // Phong weights
    float ka = 0.1f, kd = 0.7f, ks = 0.2f;
    // specular reflection term
    int alpha = 100;
    // cosine of the angle between the light director and normal to location
    float cosTheta = glm::dot(glm::normalize(glm::vec3(gradient.dir * gradient.magnitude)), glm::normalize(L));
    // refletion of light on surface
    glm::vec3 reflection = glm::normalize(L) - 2 * (glm::dot(glm::normalize(L), glm::normalize(gradient.dir))) * glm::normalize(gradient.dir);
    // cosine of the angle between the light reflection and visualization direction
    float cosPhi = glm::dot(glm::normalize(reflection), glm::normalize(V));

    glm::vec3 ambient = ka * (color * lighColor);
    glm::vec3 diffuse = kd * (color * lighColor) * cosTheta;
    glm::vec3 specular = ks * (color * lighColor) * (std::powf(cosPhi, alpha));

    glm::vec3 phongReflection = ambient + diffuse + specular;

    if (isnan(phongReflection[0]))
        return color; // return glm::vec3{ 0.0f, 0.0f, 0.0f };

    if (phongReflection[0] < 0)
        phongReflection[0] = 0;
    else if (phongReflection[0] > 1)
        phongReflection[0] = 1;
    if (phongReflection[1] < 0)
        phongReflection[1] = 0;
    else if (phongReflection[1] > 1)
        phongReflection[1] = 1;
    if (phongReflection[2] < 0)
        phongReflection[2] = 0;
    else if (phongReflection[2] > 1)
        phongReflection[2] = 1;

    return phongReflection;
}


// Compute Phong Shading with warm to cool color, given the voxel color (material color), the gradient, the light vector , view vector,
// the ro and beta that affect the blending, the warm color and the cool color.
glm::vec3 Renderer::computePhongShadingWarm2Cool(const glm::vec3& color, const volume::GradientVoxel& gradient, const glm::vec3& L, const glm::vec3& V, float ro, float beta, const glm::vec4 warmColor, const glm::vec4 coolColor)
{
    //Warm and Cool Color from GUI
    glm::vec3 lightColorCool = coolColor;
    glm::vec3 lightColorWarm = warmColor;
    //Phong weights
    float ka = 0.1f, kd = 0.7f;

    //cosine of the angle between the light directon from above and normal to location
    float cosTheta1 = glm::dot(glm::normalize(glm::vec3(gradient.dir * gradient.magnitude)), glm::normalize(L));
    // cosine of the angle between the light directon from below and normal to location
    float cosTheta2 = glm::dot(glm::normalize(glm::vec3(gradient.dir * gradient.magnitude)), glm::normalize(-L));

    //calculate k constants according to parameters from GUI
    glm::vec3 kcool = lightColorCool + ro * kd;
    glm::vec3 kwarm = lightColorWarm + beta * kd;

    lightColorCool = (kwarm - kcool) / 2.0f;
    lightColorWarm = (kcool - kwarm) / 2.0f;

    glm::vec3 ambientLight = (kcool + kwarm) / 2.0f;

    glm::vec3 diffuseCool = kd * (color * lightColorCool) * cosTheta1;
    glm::vec3 diffuseWarm = kd * (color * lightColorWarm) * cosTheta2;

    glm::vec3 phongReflection = ambientLight + diffuseCool + diffuseWarm;

    if (isnan(phongReflection[0])) return color; //return glm::vec3{ 0.0f, 0.0f, 0.0f };

    if (phongReflection[0] < 0) phongReflection[0] = 0;
    else if (phongReflection[0] > 1) phongReflection[0] = 1;
    if (phongReflection[1] < 0) phongReflection[1] = 0;
    else if (phongReflection[1] > 1) phongReflection[1] = 1;
    if (phongReflection[2] < 0) phongReflection[2] = 0;
    else if (phongReflection[2] > 1) phongReflection[2] = 1;

    return phongReflection;
}

// ======= TODO: IMPLEMENT ========
// In this function, implement 1D transfer function raycasting.
// Use getTFValue to compute the color for a given volume value according to the 1D transfer function.
glm::vec4 Renderer::traceRayComposite(const Ray& ray, float sampleStep) const
{
    // Incrementing samplePos directly instead of recomputing it each frame gives a measureable speed-up.
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;
    glm::vec4 composite = {0.0f, 0.0f, 0.0f, 0.0f}, C = {};

    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {

        // Get color from transfer function
        glm::vec4 val = getTFValue(m_pVolume->getSampleInterpolate(samplePos));
        // If the volume shading is on
        if (m_config.volumeShading) {
            glm::vec3 lighDirection = samplePos - m_pCamera->position();
            glm::vec3 color = { val[0], val[1], val[2] };
            glm::vec3 shade = computePhongShading(color, m_pGradientVolume->getGradientInterpolate(samplePos), lighDirection, lighDirection);
            //std::cout << "color before " << color[0] << "/" << color[1] << "/" << color[2] << " color after " << shade[0] << "/" << shade[1] << "/" << shade[2] << "\n";
            float A_ = composite[3] + (1 - composite[3]) * val[3];
            C = { shade * val[3], val[3] };
            composite = composite + (1 - composite[3]) * C;
            composite[3] = A_;

        }
        else {
            glm::vec3 shade = { val[0], val[1], val[2] };
            float A_ = composite[3] + (1 - composite[3]) * val[3];
            C = { shade * val[3], val[3] };
            composite = composite + (1 - composite[3]) * C;
            composite[3] = A_;
        }
     
    }

    return composite;
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// Looks up the color+opacity corresponding to the given volume value from the 1D tranfer function LUT (m_config.tfColorMap).
// The value will initially range from (m_config.tfColorMapIndexStart) to (m_config.tfColorMapIndexStart + m_config.tfColorMapIndexRange) .
glm::vec4 Renderer::getTFValue(float val) const
{
    // Map value from [m_config.tfColorMapIndexStart, m_config.tfColorMapIndexStart + m_config.tfColorMapIndexRange) to [0, 1) .
    const float range01 = (val - m_config.tfColorMapIndexStart) / m_config.tfColorMapIndexRange;
    const size_t i = std::min(static_cast<size_t>(range01 * static_cast<float>(m_config.tfColorMap.size())), m_config.tfColorMap.size() - 1);
    return m_config.tfColorMap[i];
}

// ======= TODO: IMPLEMENT ========
// In this function, implement 2D transfer function raycasting.
// Use the getTF2DOpacity function that you implemented to compute the opacity according to the 2D transfer function.
glm::vec4 Renderer::traceRayTF2D(const Ray& ray, float sampleStep) const
{
    return glm::vec4(0.0f);
}

// ======= TODO: IMPLEMENT ========
// This function should return an opacity value for the given intensity and gradient according to the 2D transfer function.
// Calculate whether the values are within the radius/intensity triangle defined in the 2D transfer function widget.
// If so: return a tent weighting as described in the assignment
// Otherwise: return 0.0f
//
// The 2D transfer function settings can be accessed through m_config.TF2DIntensity and m_config.TF2DRadius.
float Renderer::getTF2DOpacity(float intensity, float gradientMagnitude) const
{
    return 0.0f;
}

// This function computes if a ray intersects with the axis-aligned bounding box around the volume.
// If the ray intersects then tmin/tmax are set to the distance at which the ray hits/exists the
// volume and true is returned. If the ray misses the volume the the function returns false.
//
// If you are interested you can learn about it at.
// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
bool Renderer::instersectRayVolumeBounds(Ray& ray, const Bounds& bounds) const
{
    const glm::vec3 invDir = 1.0f / ray.direction;
    const glm::bvec3 sign = glm::lessThan(invDir, glm::vec3(0.0f));

    float tmin = (bounds.lowerUpper[sign[0]].x - ray.origin.x) * invDir.x;
    float tmax = (bounds.lowerUpper[!sign[0]].x - ray.origin.x) * invDir.x;
    const float tymin = (bounds.lowerUpper[sign[1]].y - ray.origin.y) * invDir.y;
    const float tymax = (bounds.lowerUpper[!sign[1]].y - ray.origin.y) * invDir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;
    tmin = std::max(tmin, tymin);
    tmax = std::min(tmax, tymax);

    const float tzmin = (bounds.lowerUpper[sign[2]].z - ray.origin.z) * invDir.z;
    const float tzmax = (bounds.lowerUpper[!sign[2]].z - ray.origin.z) * invDir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    ray.tmin = std::max(tmin, tzmin);
    ray.tmax = std::min(tmax, tzmax);
    return true;
}

// This function inserts a color into the framebuffer at position x,y
void Renderer::fillColor(int x, int y, const glm::vec4& color)
{
    const size_t index = static_cast<size_t>(m_config.renderResolution.x * y + x);
    m_frameBuffer[index] = color;
}
}