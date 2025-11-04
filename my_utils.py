import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from utils import normalize, cosine

def load_image(img_path):
    return cv2.imread(img_path)

def show_images(imgs, titles):
    assert len(imgs) == len(titles)

    plt.figure(figsize=(15, 5))
    
    sub_plots_num = len(imgs)
    
    for i in range(sub_plots_num):
        plt.subplot(100 + 10 * sub_plots_num + i + 1)
        plt.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def show_annotated_lines(image, annotations, num_pairs=2):
        
    points = annotations
        
    colors_bgr = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)] 
    
    for i in range(0, num_pairs * 4, 2):
        pt1 = points[i]
        pt2 = points[i + 1]
        
        pt1_int = (int(pt1[0]), int(pt1[1]))
        pt2_int = (int(pt2[0]), int(pt2[1]))
        
        line_index = i // 2
        pair_index = line_index // 2 
        
        color = colors_bgr[pair_index]
        
        cv2.line(image, pt1_int, pt2_int, color=color, thickness=3)

    show_images([image], ['Annotated image'])
    
def to_homo(p):
    return np.array([p[0], p[1], 1])

def get_affine_rectification_matrix(annotations):
    points = [to_homo(annotations[i]) for i in range(8)] 
    lines = [np.cross(points[i], points[i + 1]) for i in range(0, 8, 2)] 
    
    p1 = np.cross(lines[0], lines[1])
    p2 = np.cross(lines[2], lines[3])
    
    l_inf_dash = normalize(np.cross(p1, p2))
    
    H_affine = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [l_inf_dash[0], l_inf_dash[1], l_inf_dash[2]]
    ])
    
    return H_affine
    
def get_metric_rectification_matrix(annotations):
    points = [to_homo(annotations[i]) for i in range(8)] 
    lines = [np.cross(points[i], points[i + 1]) for i in range(0, 8, 2)] 
    
    def orthogonality_constraint(l, m):
        """Returns coefficients [a_coef, b_coef, c_coef] for l^T C* m = 0"""
        l1, l2, l3 = l
        m1, m2, m3 = m
        return np.array([l1*m1, l1*m2 + l2*m1, l2*m2], dtype=float)
    
    A = np.vstack([
        orthogonality_constraint(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)
    ])
    
    U_svd, D_svd, Vt = np.linalg.svd(A)
    s = Vt[-1, :]
    
    s = s / s[-1]
    a, b, c = s
    
    Cstar = np.array([
        [a, b, 0],
        [b, c, 0],
        [0, 0, 0]
    ], dtype=float)
    
    U, D, Vt = np.linalg.svd(Cstar)
    H = np.diag([1/np.sqrt(D[0]), 1/np.sqrt(D[1]), 1]) @ U.T 
    
    H = H / H[2, 2]

    return H

def get_metric_rectification_matrix_with_affine_distortion(annotations, num_pairs=5):
    points = [to_homo(annotations[i]) for i in range(num_pairs * 4)] 
    lines = [normalize(np.cross(points[i], points[i + 1])) for i in range(0, num_pairs * 4, 2)] 
    
    def orthogonality_constraint(l, m):
        l1, l2, l3 = l
        m1, m2, m3 = m
        # This represents l^T * C* * m = 0 where C* = [a b d; b c e; d e f]
        return np.array([
            l1*m1,
            l1*m2 + l2*m1,
            l2*m2,
            l1*m3 + l3*m1,
            l2*m3 + l3*m2,
            l3*m3
        ], dtype=float)
    
    # Build A
    A = np.vstack([
        orthogonality_constraint(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)
    ])
    
    # Solve homogeneous system A s = 0
    _, _, Vt = np.linalg.svd(A)
    
    # The issue: need to find which singular vector gives positive definite C*
    # Start from the smallest singular values
    for idx in range(Vt.shape[0] - 1, -1, -1):
        for sign in [1, -1]:
            s = sign * Vt[idx, :]
            a, b, c, d, e, f = s
            
            Cstar = np.array([[a, b, d], [b, c, e], [d, e, f]], dtype=float)
            
            # Check positive definite
            try:
                eigvals = np.linalg.eigvalsh(Cstar)
                if np.all(eigvals > 1e-10):
                    # Normalize by bottom-right element
                    Cstar = Cstar / Cstar[2, 2]
                    
                    # Cholesky decomposition (more stable than SVD for PSD matrices)
                    try:
                        L = np.linalg.cholesky(Cstar)
                        H = np.linalg.inv(L.T)
                        H = H / H[2, 2]
                        return H
                    except np.linalg.LinAlgError:
                        continue
            except:
                continue
    
    # No valid solution found
    print("Warning: Could not find positive definite solution")
    return np.eye(3)
    
def verfiy_angles(H, annotation, num_pairs=2):
    H_inv_T = np.linalg.inv(H).T
    
    points = [to_homo(annotation[i]) for i in range(num_pairs * 4)] 
    lines = [np.cross(points[i], points[i + 1]) for i in range(0, num_pairs * 4, 2)] 

    for i in range(0, len(lines), 2):
        l = lines[i]
        m = lines[i + 1]
        
        l_after = H_inv_T @ l
        m_after = H_inv_T @ m

        print(f"Pairs number {i/2 + 1}:")
        print(f"Before: {cosine(l, m)}")
        print(f"After: {cosine(l_after, m_after)}")

        print("==========")
