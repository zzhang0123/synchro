(* Algebraic comparison with Pandya et al., arXiv:1602.08749v1, Eqs. (3)-(11).
   This checks the printed expressions, not their physical sign convention.
   Run separately: wolframscript -file derivation/compare_pandya.wls *)
ClearAll["Global`*"];
root = DirectoryName[$InputFileName]; results = {};
check[name_, residual_, assumptions_: True] := Module[{r},
  r = FullSimplify[residual, Assumptions -> assumptions];
  AppendTo[results, <|"name" -> name, "pass" -> And @@ (# === 0 & /@ Flatten[{r}]),
    "residual" -> ToString[r, InputForm]|>];
  Print[If[Last[results]["pass"], "PASS ", "FAIL "], name];
];
(* Paper xi=our alpha, paper n=our m; paper nu_c is NONrelativistic
   cyclotron frequency, so nu_c=gamma omegaB/(2pi), not our continuum nu_c. *)
dop = 1 - beta Cos[alpha] Cos[theta];
frequencyMap = {cyclotron -> gamma omegaB/(2 Pi), nu -> omega/(2 Pi)};
resonance = m cyclotron/gamma - nu dop;                (* their Eq. 10 *)
argument = nu gamma beta Sin[theta] Sin[alpha]/cyclotron; (* their Eq. 11 *)
check["resonance mapping", (resonance /. frequencyMap) - (m omegaB - omega dop)/(2 Pi)];
check["Bessel argument on the line", (argument /. frequencyMap /. omega -> m omegaB/dop)
  - m beta Sin[alpha] Sin[theta]/dop];
mm = (Cos[theta] - beta Cos[alpha])/Sin[theta]; nn = beta Sin[alpha];
(* a,b stand for J_m and J'_m at the common on-line argument. *)
pandya = {mm^2 a^2 + nn^2 b^2, mm^2 a^2 - nn^2 b^2, 0, -2 mm nn a b};
manuscript = {mm^2 a^2 + nn^2 b^2, mm^2 a^2 - nn^2 b^2, 0, 2 mm nn a b};
check["I Q U kernels match", Take[manuscript - pandya, 3]];
check["V kernels have opposite signs", manuscript[[4]] + pandya[[4]]];
(* Authors' official symphony correction, commit f63fc87 (2017-09-10),
   src/integrator/integrands.c: polarization_term changes -2 M N J J' to +2. *)
corrected = ReplacePart[pandya, 4 -> 2 mm nn a b];
check["corrected symphony kernels all match", manuscript - corrected];
(* Check a nonzero V case so the previous identity is not an equality of zeros. *)
example = {beta -> 1/2, alpha -> Pi/2, theta -> Pi/3, a -> 1, b -> 1};
check["V difference is nonzero in general", (manuscript[[4]] - pandya[[4]] /. example) - 2/Sqrt[3]];
(* Eq. 3 is per unit nu; integrate its spectral line before comparing
   with manuscript C_m^2. No free factor is introduced. *)
lineIntegral = Integrate[nu^2 DiracDelta[line - d nu], {nu, 0, Infinity},
  Assumptions -> d > 0 && line > 0];
pandyaLinePrefactor = FullSimplify[2 Pi charge^2/light lineIntegral /.
  {line -> m omegaB/(2 Pi), d -> dop}];
ourPrefactor = charge^2 m^2 omegaB^2/(2 Pi light dop^3);
check["absolute line-power prefactor matches", pandyaLinePrefactor - ourPrefactor];
Export[FileNameJoin[{root, "pandya_comparison.json"}],
  <|"source" -> "https://arxiv.org/pdf/1602.08749v1", "sourceEquations" -> "3-11",
    "wolframVersion" -> $Version, "scriptSHA256" -> FileHash[$InputFileName, "SHA256", "HexString"],
    "correctionSource" -> "https://github.com/AFD-Illinois/symphony/commit/f63fc87ce2e6a3403a767a3eb113a1904698f343",
    "checks" -> results, "conclusion" -> "Printed I,Q,U and line-power normalization match; printed V is opposite. All four kernels match the authors' 2017 sign correction. Physical handedness is checked separately in check_handedness.wls.",
    "allPassed" -> And @@ Lookup[results, "pass"]|>, "RawJSON"];
Exit[If[And @@ Lookup[results, "pass"], 0, 1]];
