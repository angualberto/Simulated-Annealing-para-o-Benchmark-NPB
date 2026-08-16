module sp_solver
    use omp_lib
    implicit none
    private
    public :: factor_penta, solve_x, solve_y, solve_z, compute_b

contains

    ! Fatoracao LU (banda pentadiagonal) - feita uma unica vez
    subroutine factor_penta(l1, l2, u0, u1, u2, n)
        integer, intent(in) :: n
        real(8), intent(out) :: l1(n), l2(n), u0(n), u1(n), u2(n)
        real(8), parameter :: a = -0.25d0, b = -1.0d0, c = 6.0d0, d = -1.0d0, e = -0.25d0
        integer :: i

        u0(1) = c
        u1(1) = d
        u2(1) = e
        l1(2) = b / u0(1)
        u0(2) = c - l1(2) * u1(1)
        u1(2) = d - l1(2) * u2(1)
        u2(2) = e
        l2(3) = a / u0(1)
        l1(3) = (b - l2(3) * u1(1)) / u0(2)
        u0(3) = c - l2(3) * u2(1) - l1(3) * u1(2)
        u1(3) = d - l1(3) * u2(2)
        u2(3) = e
        do i = 4, n
            l2(i) = a / u0(i-2)
            l1(i) = (b - l2(i) * u1(i-2)) / u0(i-1)
            u0(i) = c - l2(i) * u2(i-2) - l1(i) * u1(i-1)
            u1(i) = d - l1(i) * u2(i-1)
            u2(i) = e
        end do
    end subroutine factor_penta

    ! RHS: operador difusivo 7-pontos (SP-like)
    subroutine compute_b(u, b, n)
        integer, intent(in) :: n
        real(8), intent(in) :: u(n, n, n)
        real(8), intent(out) :: b(n, n, n)
        integer :: i, j, k

        !$omp parallel do
        do k = 1, n
            do j = 1, n
                do i = 1, n
                    b(i, j, k) = 6.0d0 * u(i, j, k) &
                        - u(min(n, i+1), j, k) - u(max(1, i-1), j, k) &
                        - u(i, min(n, j+1), k) - u(i, max(1, j-1), k) &
                        - u(i, j, min(n, k+1)) - u(i, j, max(1, k-1))
                end do
            end do
        end do
    end subroutine compute_b

    ! Varredura em x: resolve A*u = b em cada linha (j,k)
    subroutine solve_x(b, u, n, l1, l2, u0, u1, u2)
        integer, intent(in) :: n
        real(8), intent(in) :: b(n, n, n), l1(n), l2(n), u0(n), u1(n), u2(n)
        real(8), intent(out) :: u(n, n, n)
        real(8) :: y(n)
        integer :: i, j, k

        !$omp parallel do private(j, k, y)
        do k = 1, n
            do j = 1, n
                y(1) = b(1, j, k)
                y(2) = b(2, j, k) - l1(2) * y(1)
                do i = 3, n
                    y(i) = b(i, j, k) - l1(i) * y(i-1) - l2(i) * y(i-2)
                end do
                u(n, j, k) = y(n) / u0(n)
                u(n-1, j, k) = (y(n-1) - u1(n-1) * u(n, j, k)) / u0(n-1)
                u(n-2, j, k) = (y(n-2) - u1(n-2) * u(n-1, j, k) - u2(n-2) * u(n, j, k)) / u0(n-2)
                do i = n-3, 1, -1
                    u(i, j, k) = (y(i) - u1(i) * u(i+1, j, k) - u2(i) * u(i+2, j, k)) / u0(i)
                end do
            end do
        end do
    end subroutine solve_x

    ! Varredura em y: resolve A*u = b em cada linha (i,k)
    subroutine solve_y(b, u, n, l1, l2, u0, u1, u2)
        integer, intent(in) :: n
        real(8), intent(in) :: b(n, n, n), l1(n), l2(n), u0(n), u1(n), u2(n)
        real(8), intent(out) :: u(n, n, n)
        real(8) :: y(n)
        integer :: i, j, k

        !$omp parallel do private(i, k, y)
        do k = 1, n
            do i = 1, n
                y(1) = b(i, 1, k)
                y(2) = b(i, 2, k) - l1(2) * y(1)
                do j = 3, n
                    y(j) = b(i, j, k) - l1(j) * y(j-1) - l2(j) * y(j-2)
                end do
                u(i, n, k) = y(n) / u0(n)
                u(i, n-1, k) = (y(n-1) - u1(n-1) * u(i, n, k)) / u0(n-1)
                u(i, n-2, k) = (y(n-2) - u1(n-2) * u(i, n-1, k) - u2(n-2) * u(i, n, k)) / u0(n-2)
                do j = n-3, 1, -1
                    u(i, j, k) = (y(j) - u1(j) * u(i, j+1, k) - u2(j) * u(i, j+2, k)) / u0(j)
                end do
            end do
        end do
    end subroutine solve_y

    ! Varredura em z: resolve A*u = b em cada linha (i,j)
    subroutine solve_z(b, u, n, l1, l2, u0, u1, u2)
        integer, intent(in) :: n
        real(8), intent(in) :: b(n, n, n), l1(n), l2(n), u0(n), u1(n), u2(n)
        real(8), intent(out) :: u(n, n, n)
        real(8) :: y(n)
        integer :: i, j, k

        !$omp parallel do private(i, j, y)
        do j = 1, n
            do i = 1, n
                y(1) = b(i, j, 1)
                y(2) = b(i, j, 2) - l1(2) * y(1)
                do k = 3, n
                    y(k) = b(i, j, k) - l1(k) * y(k-1) - l2(k) * y(k-2)
                end do
                u(i, j, n) = y(n) / u0(n)
                u(i, j, n-1) = (y(n-1) - u1(n-1) * u(i, j, n)) / u0(n-1)
                u(i, j, n-2) = (y(n-2) - u1(n-2) * u(i, j, n-1) - u2(n-2) * u(i, j, n)) / u0(n-2)
                do k = n-3, 1, -1
                    u(i, j, k) = (y(k) - u1(k) * u(i, j, k+1) - u2(k) * u(i, j, k+2)) / u0(k)
                end do
            end do
        end do
    end subroutine solve_z

end module sp_solver

program sp_class_e
    use sp_solver
    use omp_lib
    implicit none

    ! Classe E do NPB-SP: 1020 x 1020 x 1020
    integer, parameter :: N = 1020
    integer, parameter :: NITER = 400
    real(8), allocatable :: u(:, :, :), b(:, :, :)
    real(8), allocatable :: l1(:), l2(:), u0(:), u1(:), u2(:)
    integer :: it, i, j, k
    real(8) :: t0, t1, sum_u, gflops

    allocate(u(N, N, N), b(N, N, N))
    allocate(l1(N), l2(N), u0(N), u1(N), u2(N))

    call factor_penta(l1, l2, u0, u1, u2, N)

    t0 = omp_get_wtime()

    ! Condicao inicial
    !$omp parallel do
    do k = 1, N
        do j = 1, N
            do i = 1, N
                u(i, j, k) = sin(0.001d0 * dble(i)) * cos(0.001d0 * dble(j)) &
                           + 0.5d0 * sin(0.001d0 * dble(k))
            end do
        end do
    end do

    write(*, '(A, I0, A, I0, A, I0, A, I0, A)') &
        "SP-like Classe E | Malha: ", N, "x", N, "x", N, " (", N*N*N, " pts) | Iters: ", NITER

    do it = 1, NITER
        call compute_b(u, b, N)
        call solve_x(b, u, N, l1, l2, u0, u1, u2)
        call compute_b(u, b, N)
        call solve_y(b, u, N, l1, l2, u0, u1, u2)
        call compute_b(u, b, N)
        call solve_z(b, u, N, l1, l2, u0, u1, u2)

        if (mod(it, 50) == 0) then
            t1 = omp_get_wtime()
            write(*, '(A, I5, A, F12.2, A)') "Iteracao ", it, " | tempo: ", t1 - t0, " s"
        end if
    end do

    sum_u = 0.0d0
    !$omp parallel do reduction(+:sum_u)
    do k = 1, N
        do j = 1, N
            do i = 1, N
                sum_u = sum_u + u(i, j, k)
            end do
        end do
    end do

    t1 = omp_get_wtime()
    gflops = (dble(NITER) * dble(N*N*N) * 40.0d0) / 1.0d9
    write(*, '(A, F12.4)') "Tempo total (s): ", t1 - t0
    write(*, '(A, F12.6, A)') "Checksum (soma de u): ", sum_u, "  [referencia p/ reproducibilidade]"
    write(*, '(A, F10.2, A)') "Rendimento aproximado: ", gflops / (t1 - t0), " GFlops"

    deallocate(u, b, l1, l2, u0, u1, u2)
end program sp_class_e
