## Introduction
Both animation and robotics heavily rely on optimization algorithms. In order to understand what happens inside the optimizer, and to be able to debug problems efficiently, we rely on interactive visualization techniques. "Interactive" means that it is possible to change parameters during runtime and see the change in result immediately, without having to stop the application, edit, and run again. In this introductory assignment you will experiment with basic optimization and visualization techniques. The goal is to introduce you a different way of coding that is geared toward interactive applications. This will be important in future assignments.

## Task 1: Understand the code
### Part 1: Implementation of the mouse right click callback

In my implementation, when the user right-clicks on the 3D function surface, a fireworks animation is triggered. This callback generates a burst of small spheres—each moving outward in random directions from the clicked point—to simulate a fireworks display. The creative effect adds an element of fun to the exploration of the function's landscape.



https://github.com/user-attachments/assets/edba3fae-16b5-4a41-9b07-1e825b453d64



### Part 2: Visualizing Function Values on the Path

In this task, the goal was to visualize the function values along the optimization path, stored in the NumPy array Xi, which is updated with every mouse mouse. Every time a point is added, the graph should reflect the updated path.

Initially, when I did not use vedo.plot.clone2d, the plot would disappear or fail to persist correctly on the screen: 



https://github.com/user-attachments/assets/1afb5775-1206-48b8-b340-37b0c4179abd




After applying vedo.plot.clone2d, the graph correctly displayed the number of iterations on the X-axis and the corresponding function values on the Y-axis, and it remained visible as an overlay: 



https://github.com/user-attachments/assets/52e1cebb-6614-49e6-8fbe-dc592e2848bd


### Part 3: Change the Function

In this task, the goal was to allow the user to dynamically change the objective function using the graphical user interface (GUI). This was done by introducing interactivity through sliders that modify parameters of the function in real time.

Originally, the function was defined statically as:

```python
def objective(X):
    x, y = X[0], X[1]
    return np.sin(2*x*y) * np.cos(3*y)/2 + 1/2
```

I modified the implementation so that the function depends on two global parameters, sin_coeficent and cos_coeficent, which can be controlled via sliders in the GUI:

```python
def objective(X):
    global sin_coeficent, cos_coeficent
    x, y = X[0], X[1]
    return np.sin(sin_coeficent * x * y) * np.cos(cos_coeficent * y) / 2 + 1/2
```
Two sliders were added to the interface:

One controls the sin_coeficent (affecting the sinusoidal term).

The other controls the cos_coeficent (affecting the cosine term).

As the user moves the sliders, the 3D function surface updates immediately to reflect the new function values. This interactivity helps visualize how changes in the coefficients directly influence the function’s shape, frequency, and curvature.



https://github.com/user-attachments/assets/65782f20-a833-4154-8dfa-d8367b824535




### Result: ###
The function successfully updates in real time based on the slider values. This makes the interface more exploratory and educational, as users can experiment with different behaviors of the function by changing its parameters interactively.


---

## Task 2: Gradient Descent

### Description

In this task, I implemented an interactive version of the Gradient Descent optimization algorithm. The original codebase included partial functionality for optimization, which was extended and refined through the following steps:

1. **Disable path creation via mouse movement:**  
   The code previously allowed the user to draw a path by moving the mouse over the 3D function surface. This behavior was disabled to prevent interference with the controlled optimization flow. However, the underlying path logic was retained to visualize the optimization trajectory.

https://github.com/user-attachments/assets/cd0d74ca-8b45-46f3-a6ee-5762cc7b8f47

2. **Left mouse button click to set initial guess:**  
   I added a callback for left mouse button clicks. When the user clicks on the surface (or the plane under it), the path is cleared and the clicked point becomes the new initial guess for the optimization process.



https://github.com/user-attachments/assets/e278714b-6b7d-4e08-8b4b-7b1d2b5f8669




3. **Interactive Gradient Descent button and User-adjustable parameters:**  
   - A button labeled “Optimize” was added to the GUI. Each time the button is clicked, a fixed number of Gradient Descent iterations is performed, and the resulting points are appended to the path. This allows users to gradually observe the optimization trajectory.
     
https://github.com/user-attachments/assets/d79ab44c-bc28-442b-ada5-fd2a82de3c11

   - Two sliders were added:
       - One to control the **step size**.
       - Another to control the **number of iterations per button press**.

![image](https://github.com/user-attachments/assets/14cc97a5-ddcf-4d8b-8bf5-02276abf8780)

- I will demonstrate using different step sizes
  - step size = 0.5:
![image](https://github.com/user-attachments/assets/4956f8a1-a193-4029-abfa-e434899ee9c3)
  - step size = 0.35:
![image](https://github.com/user-attachments/assets/67d9e10c-5c8f-44ff-9e90-9d319ec83177)
  - step size = 0.15:
![image](https://github.com/user-attachments/assets/151213e4-445b-4426-958a-0c5a123c387f)

This setup provides a flexible and intuitive interface for observing how different parameters affect convergence behavior in Gradient Descent.

### Result

- Users can now click on a point to initialize optimization.
- Each button press performs multiple optimization steps and extends the trajectory accordingly.
- The step size and number of steps can be interactively modified using the sliders.
- The function surface and optimization path are visually updated in real time.

This task showcases the integration of interactive visualization with numerical optimization techniques, helping users understand the influence of step size and initialization in Gradient Descent.


---

## Task 3: Newton's Method

### Description

This task extends the optimization framework to support Newton’s Method and compares it directly with Gradient Descent, both visually and numerically. The following changes and enhancements were made:

1. **Maintain two separate optimization paths:**  
   The code was modified to track two independent NumPy arrays: one for the Gradient Descent path and one for the Newton's Method path, in the GUI Gradient Descent path is shown in **blue** and the Newton's Method path is shown in **red**. This allows direct visual and quantitative comparison between the two methods.

![two separate optimization paths GUI](https://github.com/user-attachments/assets/48028388-2187-48e3-abc7-33fedb7b87a9)


2. **Reuse the optimization button:**  
   The same "Optimize" button used in Task 2 was reused. Each button press now performs a fixed number of steps for both optimization methods—Gradient Descent and Newton. The number of steps and step size remain user-adjustable via sliders.

https://github.com/user-attachments/assets/27ff092f-6df6-4a9b-aeb5-96999a0c8738


3. **Overlay both paths and dot product on the graph:**  
   The 2D graph created in Task 1 was updated to display both optimization paths:
   - Gradient Descent path is shown in **blue**.
   - Newton's Method path is shown in **red**.
   - The dot product between them: `⟨∇f(x), d⟩` is shown in **green**.
   
   This side-by-side visualization allows users to observe the convergence speed and behavior differences.

![Overlay paths](https://github.com/user-attachments/assets/61eee111-9695-4804-9e94-db28f7d7091c)

4. **Comparison from different initial points:**  
   To analyze Newton’s behavior in different regions of the function landscape, I tested the algorithm starting from:
   - A point **close to a minimum** → fast and stable convergence to minimum.
![image](https://github.com/user-attachments/assets/7c8aace8-d8ce-457d-a4a8-fb9ef7b41696)

   - A point near a **saddle point** → convergence to saddle point.
![image](https://github.com/user-attachments/assets/7980dbaf-2071-4346-967f-981e2454866f)

   - A point near a **local maximum** → convergence to maximum.
![image](https://github.com/user-attachments/assets/a5180f15-d888-46e3-9de3-bdff2a50030e)
   
   These cases highlight the importance of having a **positive-definite Hessian**, which is not guaranteed near saddle points or maxima.

### Result

- The GUI clearly displays the evolution of both methods.
- The Newton path typically converges faster but is more sensitive to the initial guess and the shape of the function.
- Dot product annotations provide insight into the quality of each step direction.
- The comparison graph offers an intuitive and informative side-by-side analysis of Gradient Descent vs. Newton’s Method.

This task emphasizes the power and limitations of second-order methods and their sensitivity to curvature information, as well as the value of interactive visualization in debugging and learning.


---

## Task 4: Evaluate

### Description

This task focuses on evaluating and comparing the performance of different optimization methods, especially in terms of convergence behavior, stability, and robustness. The evaluation involved completing missing components, implementing robustness checks for Newton’s Method, and introducing stopping criteria.

1. **Line-Search Implementation:**  
   I completed the `line_search` function, which finds a suitable step size that satisfies sufficient decrease conditions (e.g., Armijo condition). This helps improve both stability and convergence rate of Gradient Descent and Newton’s Method by avoiding overly large or small steps.

2. **Handling Non-Positive-Definite Hessians:**  
   Newton’s Method can fail if the Hessian matrix is not positive-definite, as the resulting step direction may not be a descent direction. I implemented two remedies discussed in class:
   
   - **Modified Newton Method 1: Shift Hessian**  
     If the Hessian is not positive-definite, I add a multiple of the identity matrix to it:&nbsp;&nbsp;*H<sub>mod</sub> = H + λI*, where *λ > 0* is increased iteratively until *H<sub>mod</sub>* becomes positive-definite.
In the GUI Shift Hessian path is shown in **Black**.
   
   - **Modified Newton Method 2: Eigenvalue Clapping**  
     I compute the eigenvalues of the Hessian and replace any non-positive eigenvalues with a small positive threshold. The Hessian is then reconstructed to ensure positive-definiteness.
In the GUI Eigenvalue Clapping path is shown in **Yellow**.

   These methods improve Newton’s behavior near saddle points or maxima.

3. **Automatic Stopping Criteria:**  
   - Norm of the gradient:
       ```python
       if np.linalg.norm(grad) < epsilon:
           break
        ```
        Where `epsilon` is a user-defined threshold (e.g., `1e-6`). This criterion ensures that the optimization stops when the algorithm is sufficiently close to a stationary point.
    - Change in function value below a threshold
      ```python
        prev_val = func(prev_X)
        new_val = func(new_X)
        if abs(new_val - prev_val) < epsilon:
            break
        ```
      Where `new_val` is the new X func(X) value that the optimization methoed gives us, `prev_val` is the current X func(X) value and `epsilon` is a user-defined threshold (e.g., `1e-6`). This criterion ensures that the optimization stops where we are at convergence.

    The `optimize` function incorporates all of these stopping criteria.

    I also added a visual indicator in the GUI that shows, for each method, how many iterations are required to converge to the minimum; you can toggle it on and off by pressing the `z` key:
    ![image](https://github.com/user-attachments/assets/d1c5197a-cd42-4f01-aa7c-2a790fef4552)

### Evaluation Procedure:
I compared the following methods from various initialization points:

- Gradient Descent  
- Vanilla Newton’s Method  
- Newton with Regularization  
- Newton with Eigenvalue Correction
- Number of iterations to each methoed are needed to converge to a minimum
 

https://github.com/user-attachments/assets/7a572b96-194e-4933-b393-93692a87198f


### Result

- **Gradient Descent** performed reliably but required more iterations.  
- **Vanilla Newton’s Method** converged rapidly near minima but failed near saddle and maximum points due to non-descent steps.  
- **Regularized Newton** and **Eigenvalue-Corrected Newton** both successfully avoided ascent directions and ensured stability.  
- **Automatic stopping** improved usability and reproducibility by ensuring that the algorithms terminated when convergence was achieved.

This task demonstrates the importance of robustness in optimization, especially when using second-order methods. Proper handling of the Hessian and adaptive step sizing can significantly improve convergence and reliability across different initial conditions.

---

## Task 5: Numerical Derivatives

### Description

In this task, I analyzed the trade-offs between numerical and analytical computation of gradients and Hessians. While the original implementation used finite differences (FD) to approximate derivatives, this method is computationally expensive and can introduce precision errors depending on the choice of ε. To improve both speed and accuracy, I implemented analytical versions of the gradient and Hessian of the objective function.

---

### 1. Analytical Gradient and Hessian

The objective function is defined as:

```python
def objective(X):
    global sin_coeficent, cos_coeficent
    x, y = X[0], X[1]
    return np.sin(sin_coeficent * x * y) * np.cos(cos_coeficent * y) / 2 + 0.5
```
Using symbolic differentiation, the analytical gradient is:
```python
def gradient_analytic(X):
    global sin_coeficent, cos_coeficent
    x, y = X[0], X[1]
    a = sin_coeficent
    b = cos_coeficent

    dfdx = 0.5 * np.cos(a * x * y) * a * y * np.cos(b * y)
    dfdy = 0.5 * (np.cos(a * x * y) * a * x * np.cos(b * y) -
                  np.sin(a * x * y) * b * np.sin(b * y))

    return np.array([dfdx, dfdy])
```
And the analytical Hessian is:
```python
def hessian_analytic(X):
    global sin_coeficent, cos_coeficent
    x, y = X[0], X[1]
    a = sin_coeficent
    b = cos_coeficent

    # Second partial derivatives
    d2fdx2 = -0.5 * np.sin(a * x * y) * a**2 * y**2 * np.cos(b * y)

    d2fdy2 = (
        -0.5 * np.sin(a * x * y) * a**2 * x**2 * np.cos(b * y)
        - 0.5 * np.cos(a * x * y) * a * x * b * np.sin(b * y)
        - 0.5 * np.cos(a * x * y) * a * x * b * np.sin(b * y)
        - 0.5 * np.sin(a * x * y) * b**2 * np.cos(b * y)
    )

    d2fdxdy = (
        0.5 * a * np.cos(a * x * y) * np.cos(b * y)
        - 0.5 * a**2 * x * y * np.sin(a * x * y) * np.cos(b * y)
        - 0.5 * a * x * b * np.sin(a * x * y) * np.sin(b * y)
    )

    H = np.array([
        [d2fdx2, d2fdxdy],
        [d2fdxdy, d2fdy2]
    ])
    return H
```
### 2. Performance Comparison

We compare the time required to compute the gradient and Hessian numerically (using finite differences) and analytically (using the expressions above), over 1000 repetitions at the point `X = [1.0, 1.0]`.

| Method               | Time (seconds) |
|---------------------|----------------|
| Numerical Gradient  | 0.017099       |
| Analytical Gradient | 0.008025       |
| Numerical Hessian   | 0.032826       |
| Analytical Hessian  | 0.025905       |

**Result:**  
The analytical versions are faster than the numerical ones, confirming the benefit of avoiding repeated function evaluations when exact expressions are available.  
However, in scenarios where the objective function is unknown or too complex to differentiate analytically, the performance gap is less significant, and numerical methods remain a practical and reliable alternative.

---

### 3. Accuracy of Finite Differences

To assess the accuracy of the finite difference method, I computed the gradient for several `ε` values and compared the results to the analytical gradient.

#### Comparison at `X = [1.0, 1.0]`

| ε       | FD Gradient            | Error (L2-norm) |
|---------|------------------------|-----------------|
| 1e-01   | [0.409241 0.205137]    | 1.46e-02        |
| 1e-02   | [0.411955 0.219357]    | 1.48e-04        |
| 1e-03   | [0.411982 0.219501]    | 1.48e-06        |
| 1e-04   | [0.411982 0.219502]    | 1.48e-08        |
| 1e-05   | [0.411982 0.219502]    | 1.49e-10        |
| 1e-06   | [0.411982 0.219502]    | 1.52e-11        |
| 1e-07   | [0.411982 0.219502]    | 5.55e-10        |

**Result:**  
- The error decreases as `ε` becomes smaller, until a plateau is reached due to floating-point precision limits.
- The optimal `ε` balances truncation error and numerical round-off error—typically between `1e-5` and `1e-6`.

---

### Conclusion

This task demonstrated that:

- **Analytical derivatives** are both faster and more accurate than numerical approximations.
- **Finite difference methods** are sensitive to the choice of `ε`.
- For performance-critical applications, **analytical methods are preferred** when derivable.

---
