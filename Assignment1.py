#%% imports
import vedo as vd
import numpy as np
from vedo.pyplot import plot
from vedo import Latex
import random  # Import the random module for generating random directions
import time


vd.settings.default_backend= 'vtk'

#%% Callbacks
msg = vd.Text2D(pos='bottom-left', font="VictorMono") # an empty text


# Number of iterations for each optimization method when they reach minimum
method_iterations = {
    'Gradient Descent': None,
    'Vanilla Newton’s Method': None,
    'Newton with Regularization ': None,
    'Newton with Eigenvalue Clipping': None,
}

# Messages to display the number of iterations for each method
msg_gd = vd.Text2D(pos=[0.01, 0.95], c='black') # Gradient Descent message
msg_nt = vd.Text2D(pos=[0.01, 0.915], c='black') # Newton's method message  
msg_nt_ef = vd.Text2D(pos=[0.01, 0.88], c='black') # Newton's method message
msg_nt_r = vd.Text2D(pos=[0.01, 0.845], c='black') # Newton's method message

# Set the initial text for the messages
def reset_msgs():
    msg_gd.text(f"Gradient Descent number of iterations:")
    msg_nt.text(f"Newton's Method number of iterations:")
    msg_nt_ef.text(f"Newton with Eigenvalue Clipping number of iterations:")
    msg_nt_r.text(f"Newton with Regularization number of iterations:")

    for method in method_iterations:
        method_iterations[method] = None

hide_msg_flag = False  # Flag to control the visibility of messages
def hide_and_show_msgs():
    global hide_msg_flag
    if hide_msg_flag:
        msg_gd.text(f"Gradient Descent number of iterations: {method_iterations['Gradient Descent'] if method_iterations['Gradient Descent'] else ''}")
        msg_nt.text(f"Newton's Method number of iterations: {method_iterations['Vanilla Newton’s Method'] if method_iterations['Vanilla Newton’s Method'] else ''}")
        msg_nt_ef.text(f"Newton with Eigenvalue Clipping number of iterations: {method_iterations['Newton with Eigenvalue Clipping'] if method_iterations['Newton with Eigenvalue Clipping'] else ''}")
        msg_nt_r.text(f"Newton with Regularization number of iterations: {method_iterations['Newton with Regularization '] if method_iterations['Newton with Regularization '] else ''}")
        hide_msg_flag = False
    else:
        msg_gd.text("")
        msg_nt.text("")
        msg_nt_ef.text("")
        msg_nt_r.text("")
        hide_msg_flag = True
    
reset_msgs()


def OnMouseMove(evt):                ### called every time mouse moves!
    global Xi_gd , current_objective

    if evt.object is None:          # mouse hits nothing, return.
        plt.remove("Cylinder")    
        plt.remove("FlagPole") # remove the old flagpole
        plt.render() 
        return                    

    pt  = evt.picked3d               # 3d coords of point under mouse

    if pt is None:   # Add this check to ensure we have valid coordinates.
        return
        
    X  = np.array([pt[0],pt[1],current_objective([pt[0],pt[1]])])  # X = (x,y,e(x,y))

    txt =(
            f"X:  {vd.precision(X,2)}\n"
            #f"dX: {vd.precision(Xi[-1,0:2] - Xi[-2,0:2],2)}\n"
            #f"dE: {vd.precision(Xi[-1,2] - Xi[-2,2],2)}\n"
        )
        
    
    msg.text(txt)                    # update text message

    
    c = vd.Cylinder([np.append(X[0:2], 0.0), X[:]], r=0.01, c='orange5')
    plt.remove("Cylinder")    
    fp = fplt3d[0].flagpole(txt, point=X,s=0.08, c='k', font="Quikhand")
    fp.follow_camera()                 # make it always face the camera
    plt.remove("FlagPole") # remove the old flagpole
    plt.add(fp, c) # add the new flagpole and new cylinder
        
    plt.render()   # re-render the scene

# Add a 3D flag (label + pole) at a given point
def add_3d_flag(point, label=None):
    if label is None:
        label = f"({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
    flag = fplt3d[0].flagpole(label, point=point, s=0.08, c='red', font="VictorMono")
    flag.name = f"Flag_Init_Point"  #  give it a unique name
    flag.follow_camera()
    plt.add(flag)

def OnKeyPress(evt):               ### called every time a key is pressed
    global Xi_gd, Xi_newtow, Xi_newtow_EigenClip, Xi_newtow_ShiftedHessian

    if evt.keypress in ['c', 'C']: # reset Xi and the arrows
        Xi_gd = np.empty((0, 3))
        Xi_newtow = np.empty((0, 3)) # reset the path for Newton's method
        Xi_newtow_EigenClip = np.empty((0, 3)) # test the optimization functions
        Xi_newtow_ShiftedHessian = np.empty((0, 3)) # test the optimization functions  

        reset_msgs() # reset the messages

        plt.remove("Arrow").render()
        plt.remove("PathGraph")  # also remove the graph when resetting
        plt.remove("Flag_Init_Point")
        plt.render() 

    if evt.keypress in ['z', 'Z']: # hide and show the messages
        hide_and_show_msgs()  # Toggle the visibility of messages
        plt.render()  # Re-render the scene to show the changes


def OnSliderAlpha(widget, event): ### called every time the slider is moved
    val = widget.value         # get the slider value
    fplt3d[0].alpha(val)       # set the alpha (transparency) value of the surface
    fplt3d[1].alpha(val)       # set the alpha (transparency) value of the isolines

# Slider for optimization iterations
def OnSliderOptIterations(widget, event):
    global gd_iterations
    gd_iterations = int(widget.value)

def onSliderSinCoef(widget, event):  # Called every time the slider is moved
    global current_objective, fplt3d, fplt2d, Xi_gd, function_btn, Xi_newtow, Xi_newtow_EigenClip, Xi_newtow_ShiftedHessian, sin_coeficent

    sin_coeficent = widget.value  # Get the slider value

    Xi_gd = np.empty((0, 3))
    Xi_newtow = np.empty((0, 3))
    Xi_newtow_EigenClip = np.empty((0, 3)) # test the optimization functions
    Xi_newtow_ShiftedHessian = np.empty((0, 3)) # test the optimization functions

    # Update the 3D surface:
    reset_msgs() # reset the messages

    plt.remove(fplt3d)
    fplt3d = plot(lambda x, y: current_objective([x, y]), c='terrain')
    plt.remove(fplt2d)
    fplt2d = fplt3d.clone()
    fplt2d[0].lighting('off')
    fplt2d[0].vertices[:,2] = 0
    fplt2d[1].vertices[:,2] = 0
    plt.add(fplt3d)
    plt.add(fplt2d)
    plt.remove("Arrow").render()
    plt.remove("PathGraph")  # also remove the graph when resetting
    plt.remove("FlagPole") # remove the old flagpole
    plt.remove("Flag_Init_Point") # remove the old flagpole
    plt.remove("Cylinder") 
    plt.render()     

def onSliderCosCoef(widget, event):  # Called every time the slider is moved
    global current_objective, fplt3d, fplt2d, Xi_gd, function_btn, Xi_newtow, Xi_newtow_EigenClip, Xi_newtow_ShiftedHessian, cos_coeficent

    cos_coeficent = widget.value  # Get the slider value

    Xi_gd = np.empty((0, 3))
    Xi_newtow = np.empty((0, 3))
    Xi_newtow_EigenClip = np.empty((0, 3)) # test the optimization functions
    Xi_newtow_ShiftedHessian = np.empty((0, 3)) # test the optimization functions

    # Update the 3D surface:
    reset_msgs() # reset the messages

    plt.remove(fplt3d)
    fplt3d = plot(lambda x, y: current_objective([x, y]), c='terrain')
    plt.remove(fplt2d)
    fplt2d = fplt3d.clone()
    fplt2d[0].lighting('off')
    fplt2d[0].vertices[:,2] = 0
    fplt2d[1].vertices[:,2] = 0
    plt.add(fplt3d)
    plt.add(fplt2d)
    plt.remove("Arrow").render()
    plt.remove("PathGraph")  # also remove the graph when resetting
    plt.remove("FlagPole") # remove the old flagpole
    plt.remove("Flag_Init_Point") # remove the old flagpole
    plt.remove("Cylinder") 
    plt.render()     

def onSliderStepSize(widget, event):  # Called every time the slider is moved
    global step_size, Xi_gd, Xi_newtow, Xi_newtow_EigenClip, Xi_newtow_ShiftedHessian

    step_size = widget.value  # Get the slider value
 

#task 1.1
def on_right_click(evt):  # Called every time the right mouse button is clicked
    if evt.object is None:  # If no object is clicked, return
        return

    pt = evt.picked3d  # Get the 3D coordinates of the clicked point

    # Generate random directions for the fireworks
    num_particles = 20  # Number of particles in the fireworks
    directions = [
        [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)]
        for _ in range(num_particles)
    ]

    # Predefined list of colors for the fireworks
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']

    # Create spheres for the fireworks
    spheres = []
    for direction in directions:
        sphere = vd.Sphere(pt, r=0.02, c=random.choice(colors), alpha=0.9)  # Small sphere
        spheres.append((sphere, np.array(direction)))  # Store the sphere and its direction
        plt.add(sphere)  # Add the sphere to the plotter

    # Animate the spheres moving outward
    for scale in np.linspace(1, 3, 30):  # Gradually increase the distance
        for sphere, direction in spheres:
            sphere.pos(pt + scale * direction)  # Move the sphere outward
        plt.render()  # Re-render the scene to show the updated positions

    # Remove the spheres after the animation
    for sphere, _ in spheres:
        plt.remove(sphere)
    plt.render()


def drew_graph():  # Draw the path graph for the optimization process 
    #task 1.2 / task 3
    # Ensure Xi_gd and Xi_newtow are defined as (n, 3) numpy arrays
    indices_gd = np.arange(len(Xi_gd))          
    values_gd = Xi_gd[:, 2]                    

    indices_nt = np.arange(len(Xi_newtow))      
    values_nt = Xi_newtow[:, 2] 

    # Use helper to get dot products for Newton steps
    raw, dot_values_scaled = compute_dot_products(current_objective, Xi_newtow)

    indices_nt_ef = np.arange(len(Xi_newtow_EigenClip)) 
    values_nt_ef = Xi_newtow_EigenClip[:, 2] 

    indices_nt_gc = np.arange(len(Xi_newtow_ShiftedHessian))   
    values_nt_gc = Xi_newtow_ShiftedHessian[:, 2]


    # Plot both paths using vedo.plot
    path_graph = plot(indices_gd, values_gd, 
                  title='Function Values and Dot Products',
                  xtitle='Step', ytitle='Value',
                  lc='blue', ylim=(0, 1),
                  label='Gradient Descent')

    path_graph += plot(indices_nt, values_nt,
                   lc='red', like=path_graph,
                   label='Newton\'s Method')

    path_graph += plot(indices_nt, raw,
                   lc='green', lw=2, like=path_graph,
                   label='Dot Product ∇f · p')
    
    path_graph += plot(indices_nt_ef, values_nt_ef,
                   lc='yellow', like=path_graph,
                   label='Newton\'s Eigenvalue Clipping')
    
    path_graph += plot(indices_nt_gc, values_nt_gc,
                   lc='black', like=path_graph,
                   label='Newton\'s ShiftedHessian')
    

    # Add a semi-transparent legend with better sizing
    path_graph.add_legend(
        pos=[0.42, -0.2],  # Position of the legend box
        s=1,          # Slightly larger font size
        c='k',           # Text color
        alpha=0.2,       # Box transparency
        bc='white',      # Background color of legend box
        padding=0.1,     
        radius=0.01,
        lw=1,
        lc='black'
    )

    # Use clone2d() so that the graph stays as an overlay on the screen
    path_graph2d = path_graph.clone2d(pos = 'top-right')

    
    # Remove any previously drawn graph (using a unique name "PathGraph")
    plt.remove("PathGraph")
    path_graph2d.name = "PathGraph"
    plt.add(path_graph2d) # Add the new graph to the plotter

def on_left_click(evt):  # Called every time the left mouse button is clicked
    global Xi_gd, current_objective, Xi_newtow, Xi_newtow_EigenClip, Xi_newtow_ShiftedHessian
    # If user clicked outside the surface/plane, do nothing
    if evt.object is None:
        return
    
    if evt.actor == optimize_button:
        opt_btn_press(evt)
        return

    # Get the 3D coordinates of the clicked point
    pt = evt.picked3d
    if pt is None:
        return
    
    plt.remove("Arrow").render()
    plt.remove("PathGraph")  # also remove the graph when resetting
    plt.remove("Flag_Init_Point") # remove the old flagpole
    reset_msgs() # reset the messages
    plt.render() 

    
    # Clear the existing path
    Xi_gd = np.empty((0, 3))
    Xi_newtow = np.empty((0, 3))
    Xi_newtow_EigenClip = np.empty((0, 3)) 
    Xi_newtow_ShiftedHessian = np.empty((0, 3)) 

    # Use the clicked (x, y) as the new initial guess for the optimizer
    X_init = [pt[0], pt[1]]

    # Create a row with shape (1, 3)
    new_row = np.array([[pt[0], pt[1], current_objective([pt[0], pt[1]])]])

    add_3d_flag(new_row[0])  # Add a flag at the clicked point

    # Append the row to Xi
    Xi_gd = np.append(Xi_gd, new_row, axis=0)
    Xi_newtow = np.append(Xi_newtow, new_row, axis=0)
    Xi_newtow_EigenClip = np.append(Xi_newtow_EigenClip, new_row, axis=0) 
    Xi_newtow_ShiftedHessian = np.append(Xi_newtow_ShiftedHessian, new_row, axis=0)
    plt.render()
    

def opt_btn_press(evt):  # Called every time the button is pressed
    global Xi_gd, current_objective , Xi_newtow, gd_iterations, Xi_newtow_EigenClip, Xi_newtow_ShiftedHessian

    if len(Xi_gd) < 1:
        msg.text("Please click on the surface to set an initial point.")
        return
    for _ in range(gd_iterations):
        # Run optimization with the new initial guess
        X_opt,flag = optimize(current_objective, Xi_gd[-1][0:2], gradient_descent, tol=1e-6, iter_max=1)

        # Append the starting point and optimization result to Xi_gd
        start_pt = np.array([Xi_gd[-1][0], Xi_gd[-1][1], Xi_gd[-1][2]])
        end_pt   = np.array([X_opt[0], X_opt[1], current_objective(X_opt)])
        Xi_gd = np.vstack([Xi_gd, end_pt])

        print(flag)
        if flag:
            # draw an arrow to the new optimized point using gradient descent
            arrow = vd.Arrow(start_pt, end_pt, s=0.001, c='blue4')
            plt.add(arrow)
        elif not method_iterations['Gradient Descent']:
            method_iterations['Gradient Descent'] = len(Xi_gd) - 1
            if not hide_msg_flag:
                msg_gd.text(f"Gradient Descent number of iterations: {method_iterations['Gradient Descent']}")
            

        # Run optimization with the new initial guess using Newton's method
        X_opt,flag = optimize(current_objective, Xi_newtow[-1][0:2], Newton, tol=1e-6, iter_max=1)

        # Append the starting point and optimization result to Xi_newtow
        start_pt = np.array([Xi_newtow[-1][0], Xi_newtow[-1][1],Xi_newtow[-1][2]])
        end_pt   = np.array([X_opt[0], X_opt[1], current_objective(X_opt)])
        Xi_newtow = np.vstack([Xi_newtow, end_pt])

        if flag:
            # draw an arrow to the new optimized point using Newton's method
            arrow = vd.Arrow(start_pt, end_pt, s=0.001, c='red')
            plt.add(arrow)
        elif not method_iterations['Vanilla Newton’s Method']:
            method_iterations['Vanilla Newton’s Method'] = len(Xi_newtow) - 1
            if not hide_msg_flag:
                msg_nt.text(f"Newton's Method number of iterations: {method_iterations['Vanilla Newton’s Method']}")
            

        X_opt,flag = optimize(current_objective, Xi_newtow_EigenClip[-1][0:2], Modified_Newton_EigenFix, tol=1e-6, iter_max=1)

        # Append the starting point and optimization result to Xi_newtow_eigenFix
        start_pt = np.array([Xi_newtow_EigenClip[-1][0], Xi_newtow_EigenClip[-1][1], Xi_newtow_EigenClip[-1][2]])
        end_pt   = np.array([X_opt[0], X_opt[1], current_objective(X_opt)])
        Xi_newtow_EigenClip = np.vstack([Xi_newtow_EigenClip, end_pt])

        if flag:
            # draw an arrow to the new optimized point using Modified Newton's method
            arrow = vd.Arrow(start_pt, end_pt, s=0.001, c='yellow')
            plt.add(arrow)
        elif not method_iterations['Newton with Eigenvalue Clipping']:
            method_iterations['Newton with Eigenvalue Clipping'] = len(Xi_newtow_EigenClip) - 1
            if not hide_msg_flag:
                msg_nt_ef.text(f"Newton with Eigenvalue Clipping number of iterations: {method_iterations['Newton with Eigenvalue Clipping']}")

        X_opt,flag = optimize(current_objective, Xi_newtow_ShiftedHessian[-1][0:2], Modified_Newton_ShiftedHessian, tol=1e-6, iter_max=1)

        # Append the starting point and optimization result to Xi_newtow_gradCheck
        start_pt = np.array([Xi_newtow_ShiftedHessian[-1][0], Xi_newtow_ShiftedHessian[-1][1], Xi_newtow_ShiftedHessian[-1][2]])
        end_pt   = np.array([X_opt[0], X_opt[1], current_objective(X_opt)])
        Xi_newtow_ShiftedHessian = np.vstack([Xi_newtow_ShiftedHessian, end_pt])

        if flag:
            # draw an arrow to the new optimized point using Modified Newton's method
            arrow = vd.Arrow(start_pt, end_pt, s=0.001, c='black')
            plt.add(arrow)
        elif not method_iterations['Newton with Regularization ']:
            method_iterations['Newton with Regularization '] = len(Xi_newtow_ShiftedHessian) - 1
            if not hide_msg_flag:
                msg_nt_r.text(f"Newton with Regularization number of iterations: {method_iterations['Newton with Regularization ']}")

        

    drew_graph()  # Call the function to draw the graph
    plt.render()

# Create the button to use gd_btn /task 2.2:
optimize_button = vd.Button(
    opt_btn_press,
    states=("Optimize", "Optimizing..."),
    pos=(0.1, 0.55),
    c=['w'],
    bc=['db'],
    font="VictorMono",
    size=30,
    bold=True,
    italic=False,
)
optimize_button.pickable = True  # Make the button pickable
#optimize_button.top = True



#%% Optimization functions
def gradient_fd(func, X, h=0.001): # finite difference gradient
    x, y = X[0], X[1]
    gx = (func([x+h, y]) - func([x-h, y])) / (2*h)
    gy = (func([x, y+h]) - func([x, y-h])) / (2*h)
    return gx, gy

def Hessian_fd(func, X, h=0.001): # finite difference Hessian
    x, y = X[0], X[1]
    gxx = (func([x+h, y]) - 2*func([x, y]) + func([x-h, y])) / h**2
    gyy = (func([x, y+h]) - 2*func([x, y]) + func([x, y-h])) / h**2
    gxy = (func([x+h, y+h]) - func([x+h, y-h]) - func([x-h, y+h]) + func([x-h, y-h])) / (4*h**2)
    H = np.array([[gxx, gxy], [gxy, gyy]])
    return H

def gradient_descent(func, X): # compute gradient step direction
    g = gradient_fd(func, X)
    return -np.array(g)

def Newton(func, X):   # compute Newton step direction
    g = gradient_fd(func, X)
    H = Hessian_fd(func, X)
    d = -np.linalg.solve(H, np.array(g))
    return np.array([d[0],d[1]])

def line_search(func, X, d, alpha=1):
    min_alpha = 1e-4
    while func(X + alpha * d) > func(X):
        alpha *= 0.5
        if alpha < min_alpha:
            break
    return alpha

def step(func, X, search_direction_function):
    global step_size
    d = search_direction_function(func, X)
    alpha = line_search(func, X, d, step_size)
    return X + d*alpha 

def optimize(func, X, search_direction_function, tol=1e-6, iter_max=10):
    flag_improved = True

    for i in range(iter_max):
        prev_val = func(X)
        # Take a search direction step
        X = step(func, X, search_direction_function)
        new_val = func(X)

        if np.linalg.norm(gradient_fd(func, X)) < tol:
            flag_improved = False
            break

        # Check for convergence 
        if abs(new_val - prev_val) < tol:
            flag_improved = False
            break

    return X, flag_improved

def compute_dot_products(func, Xi): # Compute dot products of gradient and Newton direction
    raw_dots = []
    for x in Xi:
        g = gradient_fd(func, x[0:2])
        H = Hessian_fd(func, x[0:2])
        d = -np.linalg.solve(H, np.array(g))
        dot_product = np.dot(g, d)
        raw_dots.append(dot_product)
    raw_dots = np.array(raw_dots)
    return raw_dots, raw_dots / np.max(np.abs(raw_dots))  # Normalize the dot products

#Fix 1: Modified Newton using Eigenvalue Clipping
#Replace all negative eigenvalues of the Hessian with zero, then reconstruct the matrix.
def Modified_Newton_EigenFix(func, X):
    g = np.array(gradient_fd(func, X))
    H = Hessian_fd(func, X)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(H)

    # Clip negative eigenvalues to zero
    eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)

    # Avoid zero matrix
    if np.all(eigvals_clipped == 0):
        eigvals_clipped += 1e-6

    # Reconstruct modified Hessian
    H_clipped = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    try:
        d = -np.linalg.solve(H_clipped, g)
    except np.linalg.LinAlgError:
        d = -g

    # Normalize the step direction
    if np.linalg.norm(d) > 1:
        d = d / np.linalg.norm(d)

    return d

# Fix 2: Modified Newton using Shift Hessian by λI
# Add λI to the Hessian where λ > S-λ_min to make it positive-definite.
def Modified_Newton_ShiftedHessian(func, X):
    g = np.array(gradient_fd(func, X))
    H = Hessian_fd(func, X)

    eigvals = np.linalg.eigvalsh(H)
    lambda_min = np.min(eigvals)

    # If the smallest eigenvalue is negative, shift Hessian
    if lambda_min < 0:
        lam = abs(lambda_min) + 1e-3  # A small buffer
        H += lam * np.eye(len(H))

    try:
        d = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        d = -g  # fallback to gradient descent

    # Normalize the step direction
    if np.linalg.norm(d) > 1:
        d = d / np.linalg.norm(d)

    return d

def gradient_analytic(X): # compute the gradient analytically
    global sin_coeficent, cos_coeficent
    x, y = X[0], X[1]
    a = sin_coeficent
    b = cos_coeficent

    dfdx = 0.5 * np.cos(a * x * y) * a * y * np.cos(b * y)
    dfdy = 0.5 * (np.cos(a * x * y) * a * x * np.cos(b * y) -
                  np.sin(a * x * y) * b * np.sin(b * y))

    return np.array([dfdx, dfdy])

def hessian_analytic(X): # compute the Hessian analytically
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

def compare_gradient_hessian_timings(X_sample=[1.0, 1.0], repetitions=1000):
    global objective
    print(f"Measuring performance for point X = {X_sample} over {repetitions} repetitions.\n")

    # --- Numerical Gradient ---
    start = time.time()
    for _ in range(repetitions):
        gradient_fd(objective, X_sample)
    duration_grad_fd = time.time() - start
    print(f"Numerical Gradient:  {duration_grad_fd:.6f} seconds")

    # --- Analytical Gradient ---
    start = time.time()
    for _ in range(repetitions):
        gradient_analytic(X_sample)
    duration_grad_analytic = time.time() - start
    print(f"Analytical Gradient: {duration_grad_analytic:.6f} seconds")

    # --- Numerical Hessian ---
    start = time.time()
    for _ in range(repetitions):
        Hessian_fd(objective, X_sample)
    duration_hess_fd = time.time() - start
    print(f"Numerical Hessian:   {duration_hess_fd:.6f} seconds")

    # --- Analytical Hessian ---
    start = time.time()
    for _ in range(repetitions):
        hessian_analytic(X_sample)
    duration_hess_analytic = time.time() - start
    print(f"Analytical Hessian:  {duration_hess_analytic:.6f} seconds")

def compare_fd_gradient_to_analytic(X=[1.0, 1.0]):
    global current_objective
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    grad_true = gradient_analytic(X)

    print(f"Comparing FD gradients to analytical at X = {X}")
    print("ε\t\tFD Gradient\t\t\tError")
    print("-" * 60)

    for eps in epsilons:
        grad_fd = gradient_fd(current_objective, X, h=eps)
        grad_fd = np.array(grad_fd)  # Convert tuple to array for math
        error = np.linalg.norm(grad_fd - grad_true)
        print(f"{eps:.0e}\t{np.array2string(grad_fd, precision=6)}\t{error:.2e}")


#%% Plotting
def objective(X):
    global sin_coeficent, cos_coeficent
    x, y = X[0], X[1]
    return np.sin(sin_coeficent*x*y) * np.cos(cos_coeficent*y)/2+1/2

sin_coeficent = 2 # Coefficient for the sin function in the objective function
cos_coeficent = 3 # Coefficient for the cos function in the objective function

step_size = 0.1 # Step size for the optimization functions

# Global variable for current objective function:
current_objective = objective

# Global variable for gradient descent iterations:
gd_iterations = 1  # default

compare_gradient_hessian_timings()
print("\n")
compare_fd_gradient_to_analytic()

Xi_gd = np.empty((0, 3))# test the optimization functions
Xi_newtow = np.empty((0, 3)) # test the optimization functions
Xi_newtow_EigenClip = np.empty((0, 3)) # test the optimization functions
Xi_newtow_ShiftedHessian = np.empty((0, 3)) # test the optimization functions

plt = vd.Plotter(bg2='lightblue')  # Create the plotter
fplt3d = plot(lambda x,y: objective([x,y]), c='terrain')      # create a plot from the function e. fplt3d is a list containing surface mesh, isolines, and axis
fplt2d = fplt3d.clone()            # clone the plot to create a 2D plot


fplt2d[0].lighting('off')          # turn off lighting for the 2D plot
fplt2d[0].vertices[:,2] = 0        # set the z-coordinate of the mesh to 0
fplt2d[1].vertices[:,2] = 0        # set the z-coordinate of the isolines to 0


plt.add_callback('mouse move', OnMouseMove) # add Mouse move callback
plt.add_callback('key press', OnKeyPress) # add Keyboard callback
plt.add_callback('mouse right click', on_right_click) # add right click callback / task 1.1
plt.add_slider(OnSliderAlpha,0.,1.,1., title="Alpha") # add a slider for the alpha value of the surface
plt.add_callback('mouse left click', on_left_click) # add right click callback / task 1.1
plt.add_slider(onSliderSinCoef, 0.1, 10, value=2, title="Sin Coefficient", pos = [[0.02, 0.44], [0.2, 0.44]]) # add a slider for the sin coefficient
plt.add_slider(onSliderCosCoef, 0.1, 10, value=3, title="Cos Coefficient", pos = [[0.02, 0.34], [0.2, 0.34]]) # add a slider for the cos coefficient
plt.add(optimize_button)  # Add the button to the plotter
plt.add_slider(OnSliderOptIterations, 1, 100, value=1, title="Optimize Iterations", pos = [[0.02, 0.14], [0.2, 0.14]])  # Add a slider for Optimize descent iterations
plt.add_slider(onSliderStepSize, 0.01, 1, value=0.1, title="Step size", pos = [[0.02, 0.24], [0.20, 0.24]])  # Add a slider for Optimize descent iterations

plt.show([fplt3d, fplt2d], msg, msg_gd, msg_nt,msg_nt_r,msg_nt_ef,  __doc__, viewup='z')
plt.close()


# %%
