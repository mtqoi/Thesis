# MThornton Thesis CHANGELOG: 5th October 2020

## Summary of changes
- Included the four new additions recommended by examiners SMB and BWL.
- Adopted every typographical change and correction recommended by SMB.
- Adopted (almost) every typographical change and correction recommended by BWL.
	Unable to make $P_E$ in section 3.5 title lowercase without significant changes to LaTeX stylesheet.
	Every other recommended change was included.
- Fixed references to Lin et. al. PRX (2019). 
	Previously this paper was referred to as making a Gaussianity assumption. This has been removed and the paper is correctly represented.
- Fixed the smaller issues noted during the viva, including correct normalization factors etc.
- Several other typographical and grammatical errors fixed, including error in appearance of `\varsigma`. It has been changed to `\Upsilon` everywhere.
- Minor language corrections and changes.


### Recommendation 1: discussion of classical cryptography
Additional high-level discussion of Diffie Hellman key exchange, including an illustrative academic example, added to Section 2.1 (p36-7). This provides a concrete example of a classical cryptographic protocol and motivates the rest of the section.
I considered adding a similar discussion of RSA, but decided against it because it would distract from the section's narrative flow and make the chapter even longer.

### Recommendation 2: discussion of different methods that the eavesdropper can use
Example of an unambiguous discrimination POVM added to section 1.4 (p22).
Discussion of eavesdropping measurement strategies added to section 3.6 (p86), including short discussion of unambiguous measurements and minimum error measurements, and a mention of minimum cost measurements.
Inclusion of several new references.
Place our attack analysis strategy in context and provide motivation for the information-theoretic approach which we choose.

### Recommendation 3: connection between QSS and twin-field QKD
Introduced twin-field QKD in section 2.2.10 (p52), with its ability to allow QKD over longer channels.
Included several new references including twin-field QDS, which appeared on the arXiv only a few hours before my original Thesis submission (and therefore I was unaware).
Added twin-field QKD illustration figure, Figure 2.6 (p53).
Added discussion of the relationship between twin-field QKD and our QSS protocol to section 4.5 (p128). Discussion of the relative trust assumptions made by each protocol and the potential for a new QSS protocol relying on the same QPSK twin-field QKD setup as Barnett JOSAB (2019) paper.

### Recommendation 4: provide context for twin-photon absorption
Additional context given in section 6.2.2 (p179) referencing the classic papers on two-photon absorption in atomic gases at resonance.
Added additional analytics and argumentation following the method of Simaan and Loudon J. Phys. Math. A (1975) and (1978) for the density matrix elements in Fock basis under two-photon absorption. 
Demonstration of agreement between the new analytics and the numerics which oriignally appeared in this section.