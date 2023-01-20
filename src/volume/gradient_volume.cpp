#include "gradient_volume.h"
#include <algorithm>
#include <exception>
#include <glm/geometric.hpp>
#include <glm/vector_relational.hpp>
#include <gsl/span>

namespace volume {

// Compute the maximum magnitude from all gradient voxels
static float computeMaxMagnitude(gsl::span<const GradientVoxel> data)
{
    return std::max_element(
        std::begin(data),
        std::end(data),
        [](const GradientVoxel& lhs, const GradientVoxel& rhs) {
            return lhs.magnitude < rhs.magnitude;
        })
        ->magnitude;
}

// Compute the minimum magnitude from all gradient voxels
static float computeMinMagnitude(gsl::span<const GradientVoxel> data)
{
    return std::min_element(
        std::begin(data),
        std::end(data),
        [](const GradientVoxel& lhs, const GradientVoxel& rhs) {
            return lhs.magnitude < rhs.magnitude;
        })
        ->magnitude;
}

// Compute a gradient volume from a volume
static std::vector<GradientVoxel> computeGradientVolume(const Volume& volume)
{
    const auto dim = volume.dims();

    std::vector<GradientVoxel> out(static_cast<size_t>(dim.x * dim.y * dim.z));
    for (int z = 1; z < dim.z - 1; z++) {
        for (int y = 1; y < dim.y - 1; y++) {
            for (int x = 1; x < dim.x - 1; x++) {
                const float gx = (volume.getVoxel(x + 1, y, z) - volume.getVoxel(x - 1, y, z)) / 2.0f;
                const float gy = (volume.getVoxel(x, y + 1, z) - volume.getVoxel(x, y - 1, z)) / 2.0f;
                const float gz = (volume.getVoxel(x, y, z + 1) - volume.getVoxel(x, y, z - 1)) / 2.0f;

                const glm::vec3 v { gx, gy, gz };
                const size_t index = static_cast<size_t>(x + dim.x * (y + dim.y * z));
                out[index] = GradientVoxel { v, glm::length(v) };
            }
        }
    }
    return out;
}

GradientVolume::GradientVolume(const Volume& volume)
    : m_dim(volume.dims())
    , m_data(computeGradientVolume(volume))
    , m_minMagnitude(computeMinMagnitude(m_data))
    , m_maxMagnitude(computeMaxMagnitude(m_data))
{
}

float GradientVolume::maxMagnitude() const
{
    return m_maxMagnitude;
}

float GradientVolume::minMagnitude() const
{
    return m_minMagnitude;
}

glm::ivec3 GradientVolume::dims() const
{
    return m_dim;
}

// This function returns a gradientVoxel at coord based on the current interpolation mode.
GradientVoxel GradientVolume::getGradientInterpolate(const glm::vec3& coord) const
{
    switch (interpolationMode) {
    case InterpolationMode::NearestNeighbour: {
        return getGradientNearestNeighbor(coord);
    }
    case InterpolationMode::Linear: {
        return getGradientLinearInterpolate(coord);
    }
    case InterpolationMode::Cubic: {
        // No cubic in this case, linear is good enough for the gradient.
        return getGradientLinearInterpolate(coord);
    }
    default: {
        throw std::exception();
    }
    };
}

// This function returns the nearest neighbour given a position in the volume given by coord.
// Notice that in this framework we assume that the distance between neighbouring voxels is 1 in all directions
GradientVoxel GradientVolume::getGradientNearestNeighbor(const glm::vec3& coord) const
{
    if (glm::any(glm::lessThan(coord, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord, glm::vec3(m_dim))))
        return { glm::vec3(0.0f), 0.0f };

    auto roundToPositiveInt = [](float f) {
        return static_cast<int>(f + 0.5f);
    };

    return getGradient(roundToPositiveInt(coord.x), roundToPositiveInt(coord.y), roundToPositiveInt(coord.z));
}

// ======= TODO : IMPLEMENT ========
// Returns the trilinearly interpolated gradient at the given coordinate.
// Use the linearInterpolate function that you implemented below.
GradientVoxel GradientVolume::getGradientLinearInterpolate(const glm::vec3& coord) const
{
    int x = coord[0] - 0.5f;
    int y = coord[1] - 0.5f;
    int z = coord[2] - 0.5f;

    float xFactor = coord[0] - (float)(coord[0] - 0.5f);
    float yFactor = coord[1] - (float)(coord[1] - 0.5f);
    float zFactor = coord[2] - (float)(coord[2] - 0.5f);

    //Interpolate points with same y-coord on the sides of the cell along x
    GradientVoxel interpolationAlongX1 = linearInterpolate(getGradient(x, y, z), getGradient(x + 1, y, z), xFactor);
    GradientVoxel interpolationAlongX2 = linearInterpolate(getGradient(x, y + 1, z), getGradient(x + 1, y + 1, z), xFactor);
    GradientVoxel interpolationAlongX3 = linearInterpolate(getGradient(x, y, z + 1), getGradient(x + 1, y, z + 1), xFactor);
    GradientVoxel interpolationAlongX4 = linearInterpolate(getGradient(x, y + 1, z + 1), getGradient(x + 1, y + 1, z + 1), xFactor);

    //Interpolate  the two resulting samples along y
    GradientVoxel interpolationAlongY1 = linearInterpolate(interpolationAlongX1, interpolationAlongX2, yFactor);
    GradientVoxel interpolationAlongY2 = linearInterpolate(interpolationAlongX1, interpolationAlongX4, yFactor);

    //Interpolate along z
    GradientVoxel interpolationAlongZ = linearInterpolate(interpolationAlongY1, interpolationAlongY2, zFactor);

    return interpolationAlongZ;
}

// ======= TODO : IMPLEMENT ========
// This function should linearly interpolates the value from g0 to g1 given the factor (t).
// At t=0, linearInterpolate should return g0 and at t=1 it returns g1.
GradientVoxel GradientVolume::linearInterpolate(const GradientVoxel& g0, const GradientVoxel& g1, float factor)
{
    GradientVoxel interpolation = {};

    interpolation.dir[0] = g0.dir[0] * (1 - factor) + g1.dir[0] * factor;
    interpolation.dir[1] = g0.dir[1] * (1 - factor) + g1.dir[1] * factor;
    interpolation.dir[2] = g0.dir[2] * (1 - factor) + g1.dir[2] * factor;
    interpolation.magnitude = sqrt(pow(interpolation.dir[0], 2) + pow(interpolation.dir[1], 2) + pow(interpolation.dir[2], 2));

    return interpolation;
}

// This function returns a gradientVoxel without using interpolation
GradientVoxel GradientVolume::getGradient(int x, int y, int z) const
{
    const size_t i = static_cast<size_t>(x + m_dim.x * (y + m_dim.y * z));
    return m_data[i];
}
}