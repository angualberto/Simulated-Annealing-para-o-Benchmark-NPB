module simulated_annealing_mesh
    implicit none
    private
    public :: sa_mesh_solver, PI

    real(8), parameter :: PI = 3.14159265358979323846d0
    integer, parameter :: NUM_DIRECTIONS = 32

contains

    subroutine get_direction_vectors(radius, dx_vec, dy_vec)
        integer, intent(in) :: radius
        integer, intent(out) :: dx_vec(NUM_DIRECTIONS), dy_vec(NUM_DIRECTIONS)
        integer :: k
        real(8) :: angle

        do k = 1, NUM_DIRECTIONS
            angle = (k - 1) * (2.0d0 * PI / dble(NUM_DIRECTIONS))
            dx_vec(k) = nint(dble(radius) * cos(angle))
            dy_vec(k) = nint(dble(radius) * sin(angle))
        end do
    end subroutine get_direction_vectors

    ! Contribuição da célula (x,y) para a energia total (arestas incidentes)
    function local_energy(mesh, nx, ny, x, y) result(contrib)
        integer, intent(in) :: nx, ny, x, y
        real(8), intent(in) :: mesh(nx, ny)
        real(8) :: contrib
        contrib = 0.0d0
        if (x > 1)  contrib = contrib + abs(mesh(x, y) - mesh(x-1, y))
        if (x < nx) contrib = contrib + abs(mesh(x+1, y) - mesh(x, y))
        if (y > 1)  contrib = contrib + abs(mesh(x, y) - mesh(x, y-1))
        if (y < ny) contrib = contrib + abs(mesh(x, y+1) - mesh(x, y))
    end function local_energy

    subroutine sa_mesh_solver(nx, ny, grid_resolution_m, max_iter_in)
        integer, intent(in) :: nx, ny, max_iter_in
        real(8), intent(in) :: grid_resolution_m

        real(8), allocatable :: mesh(:, :)
        integer :: dx_32(NUM_DIRECTIONS), dy_32(NUM_DIRECTIONS)
        real(8) :: temp, temp_min, alpha, delta_e, e_current, e_before, e_after
        real(8) :: prob, r, perturb, harvest
        integer :: iter, x, y, x_new, y_new, dir_idx, radius_cells, max_iter

        radius_cells = 3
        call get_direction_vectors(radius_cells, dx_32, dy_32)

        allocate(mesh(nx, ny))

        call random_number(mesh)
        mesh = mesh * 100.0d0

        temp = 1000.0d0
        temp_min = 1.0d-4
        alpha = 0.9995d0
        max_iter = max_iter_in

        ! Energia inicial: soma das arestas (cada aresta contada 2x)
        e_current = 0.0d0
        do y = 1, ny
            do x = 1, nx
                e_current = e_current + local_energy(mesh, nx, ny, x, y)
            end do
        end do
        e_current = 0.5d0 * e_current

        do iter = 1, max_iter
            if (temp <= temp_min) exit

            ! 1. Célula aleatória
            call random_number(harvest)
            x = 1 + int(harvest * dble(nx))
            call random_number(harvest)
            y = 1 + int(harvest * dble(ny))

            ! 2. Uma das 32 direções
            call random_number(harvest)
            dir_idx = 1 + int(harvest * dble(NUM_DIRECTIONS))

            ! 3. Deslocamento
            x_new = min(max(1, x + dx_32(dir_idx)), nx)
            y_new = min(max(1, y + dy_32(dir_idx)), ny)

            ! Perturbação local
            call random_number(harvest)
            perturb = (harvest - 0.5d0) * temp

            e_before = local_energy(mesh, nx, ny, x_new, y_new)
            mesh(x_new, y_new) = mesh(x_new, y_new) + perturb
            e_after = local_energy(mesh, nx, ny, x_new, y_new)
            delta_e = e_after - e_before

            ! 4. Critério de Metropolis
            if (delta_e < 0.0d0) then
                e_current = e_current + delta_e
            else
                call random_number(r)
                prob = exp(-delta_e / temp)
                if (r < prob) then
                    e_current = e_current + delta_e
                else
                    mesh(x_new, y_new) = mesh(x_new, y_new) - perturb
                end if
            end if

            ! 5. Resfriamento
            temp = temp * alpha
        end do

        write(*, '(A, I0)') "Iterações executadas: ", iter - 1
        write(*, '(A, I0, A, I0)') "Malha: ", nx, "x", ny
        write(*, '(A, F12.4)') "Energia Final: ", e_current

        deallocate(mesh)
    end subroutine sa_mesh_solver

end module simulated_annealing_mesh

program main
    use simulated_annealing_mesh
    implicit none

    ! Malha de 11,14km x 11,14km = 124 km² (182x182 = 33.124 pontos, celulas de ~61m)
    integer, parameter :: NX = 182
    integer, parameter :: NY = 182
    real(8), parameter :: DX = 61.2d0

    write(*, '(A, F8.1, A, I0)') "SA na malha de 124 km², células de ", DX, "m, ", NX*NY, " pontos..."
    call sa_mesh_solver(NX, NY, DX, 500000)
end program main
