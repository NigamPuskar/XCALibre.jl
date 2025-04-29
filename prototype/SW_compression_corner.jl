using Revise
using XCALibre

mesh_file = pkgdir(XCALibre, "prototype/Compression_Corner_2.unv")
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# Select backend and setup hardware
backend = CPU()
# backend = CUDABackend() # ru non NVIDIA GPUs
# backend = ROCBackend() # run on AMD GPUs

hardware = set_hardware(backend=backend, workgroup=1024)
# hardware = set_hardware(backend=backend, workgroup=32) # use for GPU backends

mesh_dev = mesh # use this line to run on CPU
# mesh_dev = adapt(backend, mesh)  # Uncomment to run on GPU 

#velocity = [10.0, 0.0, 0.0]
#nu = 1.58963e-5
#Re = velocity[1]*2/nu
#k_inlet = 1.0848e-3    #0.0433
#omega_inlet = 8679.5 #115.6
#I = 0.03
#k = (1.5)*(I^2)*(1.7*2)

velocity = [450, 0.0, 0.0] #560

nu = 5.58963e-5
cp = 1005.0
gamma = 1.4
Pr = 0.7

R = 287.0 #specific gas constant for air
Tref = 300.0
a = (gamma*R*Tref)^0.5 #speed of sound at sea level
Mach = velocity[1]/a #Mach number
Re = velocity[1]*3.0/nu #Reynolds number

#Working out k_inlet
Tu = 0.16 * Re^(-1/8) #Turbulence intensity expressed as a percentage 
#Tu = 2
u_prime = (Tu/100)*velocity[1]
k_inlet = 3/2*(u_prime^2)

#Working out omega
t_ratio = 5  #Keep below 50
omega_inlet = k_inlet/(t_ratio*nu)

nut_inlet = k_inlet/omega_inlet

pressure = ((1+((gamma-1)/2)*Mach^2)^-(gamma/(gamma-1)))*101325 #inlet pressure in Pa

model = Physics(
    time = Steady(),
    fluid = Fluid{Compressible}(; nu=nu,  cp=cp, gamma=gamma, Pr=Pr),
    turbulence = RANS{KOmega}( β⁺=0.09, α1=0.52, β1=0.072, σk=0.5, σω=0.5),   #sets closure constants
    energy = Energy{SensibleEnthalpy}(Tref=Tref),
    domain = mesh_dev
    )

@assign! model momentum U (
    Dirichlet(:Inlet, velocity),
    Neumann(:Outlet, 0.0),
#    Symmetry(:Plate_1),
    Wall(:Plate, [0.0, 0.0, 0.0]),
#    Wall(:Plate_1, [0.0, 0.0, 0.0]),
#    Wall(:Plate_2, [0.0, 0.0, 0.0]),
    Neumann(:Top, 0.0),
    #Neumann(:Top_1, 0.0),
    #Neumann(:Top_2, 0.0),
)

@assign! model momentum p (
    Dirichlet(:Inlet, pressure),
#   Dirichlet(:Outlet, 101325),  # Set atmospheric pressure if subsonic
    Neumann(:Outlet, 0.0),
#    Symmetry(:Plate_1, 0.0),
    Neumann(:Plate, 0.0),
#    Neumann(:Plate_1, 0.0),
#    Neumann(:Plate_2, 0.0),
#     Wall(:Plate, [0.0, 0.0, 0.0]),
#     Wall(:Plate, 0.0),
#    Wall(:Plate_2, [0.0, 0.0, 0.0]),
#    Wall(:Plate_1, 0.0),
#    Wall(:Plate_2, 0.0),
    Neumann(:Top, 0.0),
    #Neumann(:Top_1, 0.0),
    #Neumann(:Top_2, 0.0),
)

@assign! model energy h (
    FixedTemperature(:Inlet, T=Tref, model=model.energy),
    Neumann(:Outlet, 0.0),
    FixedTemperature(:Plate, T=Tref, model=model.energy),
#    FixedTemperature(:Plate_1, T=310.0, model=model.energy),
#    FixedTemperature(:Plate_2, T=310.0, model=model.energy),
    Neumann(:Top, 0.0),
    #Neumann(:Top_1, 0.0),
    #Neumann(:Top_2, 0.0),
)

@assign! model turbulence k (       
    Dirichlet(:Inlet, k_inlet),
    Neumann(:Outlet, 0.0),
#    Symmetry(:Plate_1, 0.0), 
#    Dirichlet(:Plate_2, 1e-15),       #basically 0 
    KWallFunction(:Plate),
#    Dirichlet(:Plate, 1e-15),       #basically 0 
    Neumann(:Top, 0.0),
    #Neumann(:Top_1, 0.0),
    #Neumann(:Top_2, 0.0),
)

@assign! model turbulence omega (
    Dirichlet(:Inlet, omega_inlet),
    Neumann(:Outlet, 0.0),
#    Symmetry(:Plate_1, 0.0),
#    OmegaWallFunction(:Plate_2),
    OmegaWallFunction(:Plate),
    Neumann(:Top, 0.0),
#    Neumann(:Top_1, 0.0),
#    Neumann(:Top_2, 0.0),
)

@assign! model turbulence nut (
    Dirichlet(:Inlet, nut_inlet),
    #Neumann(:Inlet, 0.0),
    Neumann(:Outlet, 0.0),
    NutWallFunction(:Plate),
#    NutWallFunction(:Plate_1),
#    NutWallFunction(:Plate_2),
    Neumann(:Top, 0.0),
#    Neumann(:Top_1, 0.0),
#    Neumann(:Top_2, 0.0),
)

 schemes = (
    U = set_schemes(divergence = Upwind),
    p = set_schemes(), # no input provided (will use defaults)
    h = set_schemes(divergence = Upwind),
    k = set_schemes(divergence = Upwind),   #has divergence
    omega = set_schemes(divergence = Upwind)    #has divergence
    
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # Options: GmresSolver
        preconditioner = Jacobi(), # Options: NormDiagonal(), DILU(), ILU0()
        convergence = 1e-7,
        relax       = 0.5,
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # Options: CgSolver, BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # Options: NormDiagonal(), LDL() (with GmresSolver)
        convergence = 1e-7,
        relax       = 0.2,
    ),   
    h = set_solver(
        model.energy.h;
        solver      = BicgstabSolver, # Options: CgSolver, BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # Options: NormDiagonal(), LDL() (with GmresSolver)
        convergence = 1e-7,
        relax       = 0.5,
    ), 
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # Options: GmresSolver
        preconditioner = Jacobi(), # Options: NormDiagonal(), DILU(), ILU0()
        convergence = 1e-7,
        relax       = 0.5,
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = BicgstabSolver, # Options: GmresSolver
        preconditioner = Jacobi(), # Options: NormDiagonal(), DILU(), ILU0()
        convergence = 1e-7,
        relax       = 0.5,
    ),
)

runtime = set_runtime(iterations=50, time_step=1, write_interval=100)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

initialise!(model.momentum.U, velocity)    
initialise!(model.momentum.p, pressure)
initialise!(model.energy.T, Tref)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, omega_inlet)
initialise!(model.turbulence.nut, nut_inlet)

residuals = run!(model, config);

using Plots

# Extract residuals for different variables
iterations = 1:length(residuals[:Ux])  # Number of iterations

# Plot residuals
plot(iterations, residuals[:p], label="Pressure (p)")
plot!(iterations, residuals[:Ux], label="Velocity (Ux)", yaxis=:log, xlabel="Iteration number (-)", ylabel="Residual (-)")
plot!(iterations, residuals[:Uy], label="Velocity (Uy)")
#plot!(iterations, residuals[:e], label="Energy (e)")
savefig("residuals.pdf")
res_U = residuals[:Ux] 




tauw, pos = wall_shear_stress(:Plate_2, model)

x = ([pos[i][1] - 0.5 for i ∈ eachindex(pos)])


using Plots

plot(x, tauw.x.values/(0.5*velocity[1]^2)/10)
Rex = velocity[1].*x/nu
#cf = 0.0664./sqrt.(Rex)
#plot!(x, cf)

# Compute theoretical skin friction coefficient
cf_theoretical = 0.0664 ./ sqrt(Re)

# Compute calculated Cf from wall shear stress
cf_calculated = tauw.x.values / (0.5 * velocity[1]^2) / 10

# Create the plot
plot(x, cf_calculated, xlabel="x(m)", ylabel="Cf(-)",
     title="Skin friction coefficient for incompressible flow", legend=:topright)

# Overlay the theoretical Cf curve
#plot!(x, cf_theoretical, label="Theoretical Cf", linestyle=:dash)

# Set axis limits
xlims!(0, maximum(x))
ylims!(0, 0.014)