import trimesh
import numpy as np

def sample_points_on_mesh(mesh, N):
    """
    Sample points on the surface of a mesh.
    
    Parameters:
        mesh (trimesh.Trimesh): Mesh to sample points from.
        N (int): Number of points to sample.
        
    Returns:
        numpy.ndarray: Array of shape (N, 3) with sampled points.
    """
    # Sample points on the surface of the mesh
    points, face_index = trimesh.sample.sample_surface(mesh, N, seed=0)
    face_normals = mesh.face_normals[face_index]
    return points, face_normals

def sample_points_on_sphere(N, center=[0, 0, 0], radius=1.0):
    """
    Uniformly sample N points on the surface of a sphere.
    
    Parameters:
        N (int): Number of points to sample.
        
    Returns:
        numpy.ndarray: Array of shape (N, 3) with sampled points.
    """
    # Generate random angles
    phi = np.random.uniform(0, 2 * np.pi, N)  # Azimuthal angle
    cos_theta = np.random.uniform(-1, 1, N)   # Uniform in cos(theta)
    theta = np.arccos(cos_theta)              # Polar angle
    
    # Convert spherical to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1]
    z = radius * np.cos(theta) + center[2] 
    
    return np.vstack((x, y, z)).T

def sample_direction_hemisphere(normal):
    dir = sample_points_on_sphere(1)[0]
    if np.dot(normal, dir) < 0.0:
        dir = -dir
    return dir

if __name__ == "__main__":
    
    normal = np.random.randn(3)
    normal = normal / np.linalg.norm(normal)
    degrees = []
    for i in range(100000):
        dir = sample_direction_hemisphere(normal)
        degree = np.arccos(np.dot(normal, dir)) * 180 / np.pi
        degrees.append(degree)
    print(f"Mean angle: {np.mean(degrees):.2f} degrees")