%FTLE Calculate Finite-time Lyapunov exponent
function ftle_ = ftle(max_eigenvalue,timespan)

ftle_ = (.5/timespan)*log(max_eigenvalue);


