# TODO

## Order of convergence

- Find t_eq by taking the mean of the concentration along the space axis, plotting and see when it reaches equilibirium
- Take mean along both axis from t_eq to t_end
- Calculate error, which is the absolute value of the difference between mean from the previous step and L/2
- Only have current position in memory and write to disk at given interval
- Time code


(mu_Np - mu) ~ normal(0, sigma^2) / sqrt(Np)

sigma^2 = int_0^2 1/2 * (x - mu)^2 dx


eta = sqrt(2 * K)


- Make save_step independant of dt - Done



- test numerical with gaussian init pos near x = 0 vs crank-nicolson (exact)

- Read about random flight
  - "Sochastic modelling in physical oceanography"; robert j adler, peter müller, b. l. rozovskii
    - Markovian
- Reflection of particle with velocity
- Particles in the coastal ocean, chapter 4
- Wilson & Flesch, flow boundaries in random flight models, 1993