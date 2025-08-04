<a name="readme-top"></a>
<!-- ABOUT THE PROJECT -->
## About The Project

This project simulates the Raman sideband cooling (RSC) of CaF molecules in a tweezer trap following the theoratical proposal from [Caldwell et al.](https://doi.org/10.1103/PhysRevResearch.2.013251) 
and experimantal implementation from [Bao et al.](https://doi.org/10.1103/PhysRevX.14.031002)

The RSC is composed of a Ramam cooling pulse driving from $|m_N=-1,n\rangle$ to $|m_N=+1,n+\Delta n\rangle$ at a high magnetic field, 
and optical pumping back to $|m_N=+1\rangle$ state. The $n$ here denotes the vibrational state in the tweezer trap.

The transition probability of a Raman cooling pulse is calculated by the transition Rabi frequency and pulse duration. The Rabi frequency depends on 
$n$, $\Delta n$ and Lamb-Dickie parameter $\eta$. Each optical pumping step has a probability proportional to $\Omega_{OP}(n,n',\eta_{op})$ of transfering 
the population from $n$ to $n'$ with a optical pumping Lamb-Dickie parameter of $\eta_{OP}$.
A Monte-Carlo method is used to decide which state the molecule goes to.


<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- GETTING STARTED -->
## Getting Started



1. In RSC_functions.py, put in the trap frequency, Lamb-Dicke parameter of the Raman transition, optical pumping beam angle and the branching ratio to the other spin manifold.
2. Run Calculate_M_factor.ipynb to generate M_FACTOR_TABLE.npy
3. Run Optimize_sequence.ipynb and have fun

You can check the calculated Rabi frequency with RSC_op.ipynb. It should recover the figure 2 in [Yu et al.](https://doi.org/10.1103/PhysRevA.97.063423)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact
Please contact Qinshu Lyu Github/lyuqinshu if you find any issues or have any suggestions.



<p align="right">(<a href="#readme-top">back to top</a>)</p>
