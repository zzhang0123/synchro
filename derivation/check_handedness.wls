(* Independent axial circular-orbit check: signed Lorentz force -> real
   retarded electric field -> rotation sense and circular intensities.
   Units omega_B=1, c=1; electron charge=-1. Positive dimensional factors
   in E do not affect handedness. No harmonic target enters the orbit solve. *)
ClearAll["Global`*"];
root = DirectoryName[$InputFileName]; results = {};
assumptions = Element[t, Reals] && 0 < bp < 1;
check[name_, residual_] := Module[{r},
  r = FullSimplify[residual, assumptions];
  AppendTo[results, <|"name" -> name,
    "pass" -> And @@ (# === 0 & /@ Flatten[{r}]),
    "residual" -> ToString[r, InputForm]|>];
  Print[If[Last[results]["pass"], "PASS ", "FAIL "], name];
];
z = {0, 0, 1};
beta = Append[DSolveValue[{vx'[t] == -vy[t], vy'[t] == vx[t],
  vx[0] == bp, vy[0] == 0}, {vx[t], vy[t]}, t], 0];
check["electron Lorentz force", D[beta, t] + Cross[beta, z]];
field[n_] := FullSimplify[-Cross[n, Cross[n - beta, D[beta, t]]]/
  (1 - n.beta)^3, assumptions];
electric = field[z];
check["axial retarded electric field", electric - bp {-Sin[t], Cos[t], 0}];
(* The scalar below is positive for rotation in the right-handed sense
   around propagation. This uses a REAL field, independent of Fourier signs. *)
check["rotation about forward propagation", z.Cross[electric, D[electric, t]]/
  (electric.electric) - 1];
check["rotation about backward propagation", (-z).Cross[field[-z], D[field[-z], t]]/
  (field[-z].field[-z]) + 1];
fourier = FullSimplify[Integrate[Take[electric, 2] Exp[I t],
  {t, 0, 2 Pi}]/(2 Pi), assumptions];
v[j_] := -2 Im[j[[1]] Conjugate[j[[2]]]];
intensity[j_] := Total[Abs[j]^2];
check["forward V over I", v[fourier]/intensity[fourier] - 1];
backward = {fourier[[1]], -fourier[[2]]}; (* basis x,-y,-z *)
check["backward V over I in right-handed sky basis", v[backward]/intensity[backward] + 1];
right = {1, I}/Sqrt[2]; left = Conjugate[right];
check["V equals right minus left circular intensity",
  v[fourier] - (Abs[Conjugate[right].fourier]^2 - Abs[Conjugate[left].fourier]^2)];
check["conjugate Fourier convention preserves physical V",
  2 Im[Conjugate[fourier[[1]]] fourier[[2]]] - v[fourier]];
(* Reversing charge reverses the orbit; the extra overall field sign
   cancels from all Stokes parameters. *)
positron = Append[DSolveValue[{ux'[t] == uy[t], uy'[t] == -ux[t],
  ux[0] == bp, uy[0] == 0}, {ux[t], uy[t]}, t], 0];
positronField = FullSimplify[Cross[z, Cross[z - positron, D[positron, t]]], assumptions];
check["reversing charge reverses physical handedness",
  z.Cross[positronField, D[positronField, t]]/(positronField.positronField) + 1];
(* Only now compare the independently established sign with the m=1
   axial limit of the manuscript's general Bessel amplitudes. *)
aa = Cos[theta]/Sin[theta] BesselJ[1, bp Sin[theta]];
bb = bp (BesselJ[0, bp Sin[theta]] - BesselJ[2, bp Sin[theta]])/2;
axial = Limit[{aa, bb}, theta -> 0];
check["axial harmonic amplitudes", axial - {bp/2, bp/2}];
check["manuscript harmonic matches real-field V over I",
  2 (Times @@ axial)/Total[axial^2] - v[fourier]/intensity[fourier]];
Export[FileNameJoin[{root, "handedness.json"}],
  <|"wolframVersion" -> $Version,
    "scriptSHA256" -> FileHash[$InputFileName, "SHA256", "HexString"],
    "checks" -> results, "allPassed" -> And @@ Lookup[results, "pass"]|>, "RawJSON"];
Exit[If[And @@ Lookup[results, "pass"], 0, 1]];
