(* Loaded by verify.wls. Finite numerical checks, separate from symbolic proofs.
   Units e=c=R=omegaB=1 fix the scale; no scale or phase is fitted. *)
caseSpecifications = {
  {1, 6/5, Pi/4, Pi/3, 0, 0},
  {3, 2, Pi/4, Pi/3, Pi/7, Pi/5},
  {7, 2, Pi/4, 2 Pi/3, -Pi/5, Pi/3},
  {5, 2, 3 Pi/4, Pi/3, Pi/4, -Pi/6},
  {25, 5, Pi/4, Pi/3, Pi/6, Pi/8},
  {1, 2, Pi/3, 0, Pi/5, Pi/7},
  {1, 2, Pi/3, Pi, Pi/5, Pi/7},
  {3, 2, Pi/3, 0, Pi/5, Pi/7}
};
(* The exact Bessel recurrence, checked above, is regular at sin(theta)=0. *)
amplitudeRegular = (-1)^m {
  bp (Cos[theta] - bl)/(2 dop) (BesselJ[m-1, argument] + BesselJ[m+1, argument]),
  I bp Derivative[0, 1][BesselJ][m, argument]};
numericalCases = Table[Module[
  {order, lorentz, pitch, view, azimuth, initialPhase, speedValue, rules,
   direct, predicted, naturalExpected, skyExpected, directStokes, scale, error, stokesError},
  {order, lorentz, pitch, view, azimuth, initialPhase} = spec;
  speedValue = Sqrt[1 - 1/lorentz^2];
  rules = {m -> order, bp -> speedValue Sin[pitch], bl -> speedValue Cos[pitch],
    theta -> view, phi -> azimuth, phase0 -> initialPhase,
    omegaB -> 1, charge -> 1, light -> 1, radius -> 1};
  (* Fourier-transform the original vector radiation field, including dt/dt'.
     This path does not use the integrated-by-parts velocity or Bessel kernel. *)
  direct = Table[NIntegrate[
    Evaluate[N[(electric[[axis]] jacobian Exp[I lineFrequency arrival]) /. rules, 45]],
    {t, 0, 2 Pi}, WorkingPrecision -> 40, AccuracyGoal -> 27, PrecisionGoal -> 27,
    MaxRecursion -> 20]/(2 Pi), {axis, 1, 2}];
  naturalExpected = N[(amplitudeRegular Exp[-I m phase0]) /. rules, 40];
  predicted = N[(I lineFrequency rotation) /. rules, 40].naturalExpected;
  scale = Max[1, Max[Abs[predicted]]];
  error = Max[Abs[direct - predicted]]/scale;
  (* Absolute line power uses 1/(2pi D) times this source-period E coefficient. *)
  directStokes = Re[stokes[direct]]/N[(2 Pi dop) /. rules, 40];
  skyExpected = Re[stokes[N[rotation /. rules, 40].naturalExpected]] * N[prefactor /. rules, 40];
  stokesError = Max[Abs[directStokes - skyExpected]]/Max[1, Max[Abs[skyExpected]]];
  record["retarded-field Fourier case " <> ToString[spec, InputForm], error < 10^-20 && stokesError < 10^-20,
    ToString[{error, stokesError}, InputForm]];
  <|"parameters_m_gamma_alpha_theta_phi_phase0" -> ToString[spec, InputForm],
    "scaledComplexAmplitudeError" -> N[error], "scaledStokesError" -> N[stokesError],
    "directStokes" -> N[directStokes], "analyticStokes" -> N[skyExpected]|>
], {spec, caseSpecifications}];

(* A second normalisation test sums harmonics and angles, with no renormalisation.
   gamma=6/5, alpha=pi/4; compare 32 and 64 harmonics with a refined quadrature.
   These are finite convergence checks, not uniform bounds on the harmonic tail. *)
intensityAt[order_Integer, eta_?NumericQ] := Module[{bpar, bperp, d, arg, a, b},
  bpar = Sqrt[11/72]; bperp = bpar;
  d = 1 - bpar eta; arg = order bperp Sqrt[1 - eta^2]/d;
  a = bperp (eta - bpar) (BesselJ[order-1, arg] + BesselJ[order+1, arg])/(2 d);
  b = bperp (BesselJ[order-1, arg] - BesselJ[order+1, arg])/2;
  order^2 (a^2 + b^2)/d^3
];
totalAt[cutoff_, precision_] := Module[{integrand},
  integrand[eta_?NumericQ] := Sum[intensityAt[order, eta], {order, 1, cutoff}];
  NIntegrate[integrand[eta], {eta, -1, 1}, WorkingPrecision -> precision,
    AccuracyGoal -> 14, PrecisionGoal -> 14, MaxRecursion -> 15]
];
(* The 2pi azimuth integral cancels the 1/(2pi) harmonic prefactor. *)
total32 = totalAt[32, 25]; total64 = totalAt[64, 35];
larmorExpected = helicalTotal /. {charge -> 1, light -> 1, omegaB -> 1,
  lorentz -> 6/5, bp -> Sqrt[11/72]};
record["harmonic sum and angular integral vs independently integrated field",
  Abs[total64/larmorExpected - 1] < 10^-12 && Abs[(total64-total32)/larmorExpected] < 10^-10,
  ToString[{total32, total64, larmorExpected, Abs[total64/larmorExpected - 1]}, InputForm]];

(* Sensitivity controls: a wrong V sign and a missing delta Jacobian must fail
   against the original electric-field integral. These do not modify any result. *)
referenceCase = numericalCases[[2]];
reference = referenceCase["directStokes"];
wrongV = referenceCase["analyticStokes"] {1, 1, 1, -1};
wrongJacobian = referenceCase["analyticStokes"] (1 - Sqrt[3]/2 Cos[Pi/4] Cos[Pi/3]);
record["detect reversed V sign", Max[Abs[reference - wrongV]]/Max[Abs[reference]] > 10^-3];
record["detect omitted delta Jacobian", Max[Abs[reference - wrongJacobian]]/Max[Abs[reference]] > 10^-3];
