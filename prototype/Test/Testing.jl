export test_explicit_relaxation!


#explicit_relaxation
function test_explicit_relaxation!(phi, phi0, alpha, time, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    alpha_values = zeros(length(phi))    

    kernel! = test_explicit_relaxation_kernel!(backend, workgroup)
    kernel!(phi, phi0, alpha, time, alpha_values, ndrange = length(phi))
    KernelAbstractions.synchronize(backend)

    return alpha_values
end

@kernel function test_explicit_relaxation_kernel!(phi, phi0, alpha, time, alpha_values)
    i = @index(Global)
    @inbounds begin
        local_alpha = alpha
        
        #Apply adaptive alpha only in first 100 iterations and for high phi gradients
        if time <=100
            gradient = abs(phi[i] - phi0[i])
            shock_threshold = 1e-3
            if gradient > shock_threshold
                local_alpha = alpha * 0.5
            end
        end

        phi[i] = phi0[i] + local_alpha*(phi[i] - phi0[i])
        alpha_values[i] = local_alpha
    end
end


#Superbee limter
# THIS IMPLEMENTATION IS WORK IN PROGRESS AND HAS NOT BEEN TESTED FOR CORRECTNESS

struct CellBased end

limit_gradient!(method::Nothing, ∇F, F, config) = nothing

### GRADIENT LIMITER - EXPERIMENTAL

function limit_gradient!(method::CellBased, ∇F, F::ScalarField, config)
# function limit_gradient!(∇F, Ff, F::ScalarField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = F.mesh
    (; cells, cell_neighbours, cell_faces, cell_nsign, faces) = mesh

    (; x, y, z) = ∇F.result

    kernel! = _limit_gradient!(backend, workgroup)
    # kernel!(x, y, z, Ff, F, cells, cell_neighbours, cell_faces, cell_nsign, faces, ndrange=length(cells))
    kernel!(method, x, y, z, F, cells, cell_neighbours, cell_faces, cell_nsign, faces, ndrange=length(cells))
    # KernelAbstractions.synchronize(backend)
end

function limit_gradient!(method::CellBased, ∇F, F::VectorField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = F.mesh
    (; cells, cell_neighbours, cell_faces, cell_nsign, faces) = mesh

    (; xx, yx, zx) = ∇F.result
    (; xy, yy, zy) = ∇F.result
    (; xz, yz, zz) = ∇F.result

    kernel! = _limit_gradient!(backend, workgroup)
    kernel!(method, xx, yx, zx, F.x, cells, cell_neighbours, cell_faces, cell_nsign, faces, ndrange=length(cells))
    # KernelAbstractions.synchronize(backend)

    kernel! = _limit_gradient!(backend, workgroup)
    kernel!(method, xy, yy, zy, F.y, cells, cell_neighbours, cell_faces, cell_nsign, faces, ndrange=length(cells))
    # KernelAbstractions.synchronize(backend)

    kernel! = _limit_gradient!(backend, workgroup)
    kernel!(method, xz, yz, zz, F.z, cells, cell_neighbours, cell_faces, cell_nsign, faces, ndrange=length(cells))
    # KernelAbstractions.synchronize(backend)
end

# @kernel function _limit_gradient!(x, y, z, Ff, F, cells, cell_neighbours, cell_faces, cell_nsign, faces)
@kernel function _limit_gradient!(::CellBased, x, y, z, F, cells, cell_neighbours, cell_faces, cell_nsign, faces)
    cID = @index(Global)

    cell = cells[cID]
    faces_range = cell.faces_range
    phiP = F[cID]
    phiMax = phiMin = phiP
 
    for fi ∈ faces_range
        nID = cell_neighbours[fi]
        phiN = F[nID]
        
        # fID = cell_faces[fi]
        # phiN = Ff[fID]

        phiMax = max(phiN, phiMax)
        phiMin = min(phiN, phiMin)
    end

    # g0 = ∇F[cID]
    grad0 = SVector{3}(x[cID] , y[cID] , z[cID])

    cc = cell.centre
    uno = one(eltype(F[cID]))
    limiter = uno
    limiterf = uno
    for fi ∈ faces_range 
        fID = cell_faces[fi]
        nID = cell_neighbours[fi]
        face = faces[fID]
        cellN = cells[nID]
        # nID = face.ownerCells[2]
        # phiN = F[nID]
        normal = face.normal
        nsign = cell_nsign[fi]
        na = nsign*normal

        
        fc = face.centre
        nc = cellN.centre
        δϕdownwind = phiN - phiP #Actual change
        δϕupwind = (nc - cc)⋅grad0 #Predicted change
        
        if abs(δϕupwind) > 1e-15
            r = (δϕdownwind/δϕupwind)
        else
            r = 0.0
        end

        ψ = max(0.0, min(2*r, 1.0), min(r, 2.0))
        #=
        if δϕ > 0
            limiterf = min(limiter, (phiMax - phiP)/δϕ)
        elseif δϕ < 0
            limiterf = min(limiter, (phiMin - phiP)/δϕ)
        # else
        #     limiterf = uno
        end
        # limiter = min(limiterf, limiter)
        =#
        #limiter = limiterf
        grad0 *= ψ
    end
    x.values[cID] = grad0[1]
    y.values[cID] = grad0[2]
    z.values[cID] = grad0[3]
end