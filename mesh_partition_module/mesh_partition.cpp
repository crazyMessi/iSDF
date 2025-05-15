#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// 3D point structure
struct Point {
    double x, y, z;
    int index;  // Original index in the mesh

    Point() : x(0), y(0), z(0), index(-1) {}
    Point(double x, double y, double z, int idx) : x(x), y(y), z(z), index(idx) {}

    double distanceTo(const Point& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
};

// Node for KD-tree like structure
struct KDNode {
    std::vector<Point> points;
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;
    bool isLeaf;
    Point centroid;
    int seedIndex;  // Index of the seed point in the original mesh

    KDNode() : isLeaf(true), seedIndex(-1) {}
};

// Calculate centroid of a set of points
Point calculateCentroid(const std::vector<Point>& points) {
    if (points.empty()) return Point();
    
    double sumX = 0, sumY = 0, sumZ = 0;
    for (const auto& p : points) {
        sumX += p.x;
        sumY += p.y;
        sumZ += p.z;
    }
    
    return Point(sumX / points.size(), sumY / points.size(), sumZ / points.size(), -1);
}

// Find point closest to the centroid
int findClosestToCentroid(const std::vector<Point>& points, const Point& centroid) {
    if (points.empty()) return -1;
    
    double minDist = std::numeric_limits<double>::max();
    int closestIdx = -1;
    
    for (size_t i = 0; i < points.size(); ++i) {
        double dist = centroid.distanceTo(points[i]);
        if (dist < minDist) {
            minDist = dist;
            closestIdx = points[i].index;  // Use original index
        }
    }
    
    return closestIdx;
}

// Build a KD-tree like structure that balances the number of points in each leaf
std::unique_ptr<KDNode> buildBalancedTree(std::vector<Point> points, int targetPointsPerLeaf) {
    auto node = std::make_unique<KDNode>();
    node->points = points;
    
    // If points are few enough, make this a leaf node
    if (points.size() <= targetPointsPerLeaf) {
        node->isLeaf = true;
        node->centroid = calculateCentroid(points);
        node->seedIndex = findClosestToCentroid(points, node->centroid);
        return node;
    }
    
    node->isLeaf = false;
    
    // Choose the axis with the largest spread
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();
    double minZ = std::numeric_limits<double>::max();
    double maxZ = std::numeric_limits<double>::lowest();
    
    for (const auto& p : points) {
        minX = std::min(minX, p.x);
        maxX = std::max(maxX, p.x);
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
        minZ = std::min(minZ, p.z);
        maxZ = std::max(maxZ, p.z);
    }
    
    double rangeX = maxX - minX;
    double rangeY = maxY - minY;
    double rangeZ = maxZ - minZ;
    
    int axis = 0;  // 0 = x, 1 = y, 2 = z
    if (rangeY > rangeX && rangeY > rangeZ) axis = 1;
    else if (rangeZ > rangeX && rangeZ > rangeY) axis = 2;
    
    // Sort points along the chosen axis
    if (axis == 0) {
        std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
            return a.x < b.x;
        });
    } else if (axis == 1) {
        std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
            return a.y < b.y;
        });
    } else {
        std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
            return a.z < b.z;
        });
    }
    
    // Split points at median
    size_t medianIdx = points.size() / 2;
    
    std::vector<Point> leftPoints(points.begin(), points.begin() + medianIdx);
    std::vector<Point> rightPoints(points.begin() + medianIdx, points.end());
    
    // Recursively build child nodes
    node->left = buildBalancedTree(leftPoints, targetPointsPerLeaf);
    node->right = buildBalancedTree(rightPoints, targetPointsPerLeaf);
    
    return node;
}

// Collect all leaf nodes from the tree
void collectLeafNodes(const KDNode* node, std::vector<const KDNode*>& leaves) {
    if (!node) return;
    
    if (node->isLeaf) {
        leaves.push_back(node);
    } else {
        collectLeafNodes(node->left.get(), leaves);
        collectLeafNodes(node->right.get(), leaves);
    }
}

// Represent a face in the mesh
struct Face {
    int v1, v2, v3;
    
    Face(int v1, int v2, int v3) : v1(v1), v2(v2), v3(v3) {}
};

// Build adjacency list for the mesh
std::vector<std::vector<int>> buildAdjacencyList(
    int numVertices,
    const std::vector<Face>& faces
) {
    std::vector<std::vector<int>> adjacencyList(numVertices);
    
    for (const auto& face : faces) {
        adjacencyList[face.v1].push_back(face.v2);
        adjacencyList[face.v1].push_back(face.v3);
        
        adjacencyList[face.v2].push_back(face.v1);
        adjacencyList[face.v2].push_back(face.v3);
        
        adjacencyList[face.v3].push_back(face.v1);
        adjacencyList[face.v3].push_back(face.v2);
    }
    
    // Remove duplicates in each adjacency list
    for (auto& neighbors : adjacencyList) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    
    return adjacencyList;
}

// Perform BFS from seed points to assign vertices to regions
std::vector<int> partitionMeshBFS(
    int numVertices,
    const std::vector<std::vector<int>>& adjacencyList,
    const std::vector<int>& seedIndices
) {
    std::vector<int> partition(numVertices, -1);  // -1 means unassigned
    std::vector<double> minDistance(numVertices, std::numeric_limits<double>::max());
    
    // Initialize a queue for BFS
    std::queue<std::pair<int, int>> queue;  // (vertex, region)
    
    // Start from all seed points
    for (size_t i = 0; i < seedIndices.size(); ++i) {
        int seedIdx = seedIndices[i];
        std::pair<int, int> seedPair(seedIdx, i);
        queue.push(seedPair);
        partition[seedIdx] = i;
        minDistance[seedIdx] = 0;
    }
    
    // Perform BFS
    while (!queue.empty()) {
        // Get the current vertex and its region
        std::pair<int, int> current = queue.front();
        int currentVertex = current.first;
        int region = current.second;
        queue.pop();
        
        double currentDist = minDistance[currentVertex];
        
        // Process neighbors
        const std::vector<int>& neighbors = adjacencyList[currentVertex];
        for (int neighbor : neighbors) {
            double newDist = currentDist + 1;  // Using topological distance (edge count)
            
            if (newDist < minDistance[neighbor]) {
                minDistance[neighbor] = newDist;
                partition[neighbor] = region;
                std::pair<int, int> neighborPair(neighbor, region);
                queue.push(neighborPair);
            }
        }
    }
    
    return partition;
}

// Main function to process a single mesh
py::tuple processSingleMesh(
    py::array_t<double> vertices,
    py::array_t<int> faces,
    int targetPointsPerLeaf
) {
    // Get array buffer info
    py::buffer_info vertBuf = vertices.request();
    py::buffer_info faceBuf = faces.request();
    
    // Check dimensions
    if (vertBuf.ndim != 2 || vertBuf.shape[1] != 3) {
        throw std::runtime_error("Vertices must be an Nx3 array");
    }
    
    if (faceBuf.ndim != 2 || faceBuf.shape[1] != 3) {
        throw std::runtime_error("Faces must be an Mx3 array");
    }
    
    // Get pointers to data
    double* vertPtr = static_cast<double*>(vertBuf.ptr);
    int* facePtr = static_cast<int*>(faceBuf.ptr);
    
    int numVertices = static_cast<int>(vertBuf.shape[0]);
    int numFaces = static_cast<int>(faceBuf.shape[0]);
    
    // Convert to our internal format
    std::vector<Point> points;
    points.reserve(numVertices);
    
    for (int i = 0; i < numVertices; ++i) {
        points.emplace_back(
            vertPtr[i*3],     // x
            vertPtr[i*3 + 1], // y
            vertPtr[i*3 + 2], // z
            i                 // original index
        );
    }
    
    std::vector<Face> meshFaces;
    meshFaces.reserve(numFaces);
    
    for (int i = 0; i < numFaces; ++i) {
        meshFaces.emplace_back(
            facePtr[i*3],     // v1
            facePtr[i*3 + 1], // v2
            facePtr[i*3 + 2]  // v3
        );
    }
    
    // Build a balanced tree
    auto tree = buildBalancedTree(points, targetPointsPerLeaf);
    
    // Collect leaf nodes to find seed points
    std::vector<const KDNode*> leaves;
    collectLeafNodes(tree.get(), leaves);
    
    // Extract seed indices
    std::vector<int> seedIndices;
    for (const auto* leaf : leaves) {
        if (leaf->seedIndex >= 0) {
            seedIndices.push_back(leaf->seedIndex);
        }
    }
    
    // Build adjacency list for BFS
    auto adjacencyList = buildAdjacencyList(numVertices, meshFaces);
    
    // Partition the mesh using BFS to get vertex regions
    std::vector<int> vertexRegions = partitionMeshBFS(numVertices, adjacencyList, seedIndices);
    
    // Calculate triangle regions
    std::vector<int> triangleRegions(numFaces, -1);
    
    for (int i = 0; i < numFaces; ++i) {
        int v1 = facePtr[i*3];
        int v2 = facePtr[i*3 + 1];
        int v3 = facePtr[i*3 + 2];
        
        int r1 = vertexRegions[v1];
        int r2 = vertexRegions[v2];
        int r3 = vertexRegions[v3];
        
        // Rule 1: If a != b != c, use region a
        if (r1 != r2 && r2 != r3 && r1 != r3) {
            triangleRegions[i] = r1;
        }
        // Rule 2: If there are at least two vertices with the same region, use that region
        else {
            // Check for pairs with the same region
            if (r1 == r2) {
                triangleRegions[i] = r1;
            } else if (r2 == r3) {
                triangleRegions[i] = r2;
            } else if (r1 == r3) {
                triangleRegions[i] = r1;
            } else {
                // This should not happen as we already checked the case where all regions are different
                triangleRegions[i] = r1;
            }
        }
    }
    
    // Return both vertex regions and triangle regions as a tuple
    return py::make_tuple(vertexRegions, triangleRegions);
}

PYBIND11_MODULE(mesh_partition, m) {
    m.doc() = "Mesh partitioning module";
    
    m.def("process_single_mesh", &processSingleMesh, 
          "Process a single mesh and return both vertex regions and triangle regions",
          py::arg("vertices"), py::arg("faces"), py::arg("target_points_per_leaf") = 100);
} 