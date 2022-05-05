# periodic-bottlenecks


We used numerical simulations to get an idea of how the different parameters influence the phage growth and evolution, and to compare with our predictions. We also made a comparison between our semi-deterministic approximation and the fully stochastic dynamics where WT phages too are modelled as a branching process.

\paragraph{Fully stochastic simulation}
In this setup, all phage populations including the WT have a stochastic behaviour. Phages are not considered individually but are grouped by populations carrying the same set of mutations. \\
We use Gillespie's algorithm \cite{Gillespie1977} to simulate the evolution of the system: starting from time 0, we draw an exponentially distributed random variable to fix the time increment (which is also the time of the next event). Then we pick randomly the event's identity (birth, death or mutation) proportionally to the corresponding rates. Then we update the population sizes according to this event. This procedure is repeated until the time reaches the length of one cycle.\\
The dilution of phages between two cycles is then performed by picking for each population a new size given by a binomial variable with parameters (population size, dilution factor).

\paragraph{Semi-deterministic simulation}
In the semi-deterministic approximation of the previous simulation, we treat the WT population differently: its growth is deterministic, thus we only have to compute $f(t)=n^\beta e^{r_0t}$ for each time step $t$. In the following cycles, population of mutants that reach a sufficient size will also switch to the deterministic regime, while others are still updated using Gillespie's algorithm.

\paragraph{$\tau$-leaping approximation}
We also implemented faster versions of the two aforementioned algorithms by using a $\tau$-leaping approximation. The principle of this approximation was first described by Gillespie in \cite{Gillespie2001} to speed up the algorithm carrying his name. What makes these simulations slow is that when the population size increase, events happen more often and time intervals between events become smaller and smaller. Thus the optimisation proposed by Gillespie is to fix a lower limit to the time increment. When rates become too high, we use a fixed time increment and simulate Poisson random variables to determine how many of each events happened during that interval. If after updating all population sizes one of them is negative, then we divide the increment by 2 and redraw the Poisson variables.
