(* Independent classical single-electron derivation. Run with wolframscript -file.
   No Python imports, fitted constants, or replacements by the target formulas.
   Starting laws: Lorentz force, retarded far-zone electric field, Poynting flux.
   Fourier transform: exp(+I omega t); real-field reconstruction: exp(-I omega t).
   e>0, electron charge -e; n points from source to observer. *)
ClearAll["Global`*"];
$HistoryLength = 0;
root = DirectoryName[$InputFileName];
checks = {}; expressions = <||>;
record[name_, ok_, detail_: ""] := (
  AppendTo[checks, <|"name" -> name, "pass" -> TrueQ[ok], "detail" -> detail|>];
  Print[If[TrueQ[ok], "PASS ", "FAIL "], name];
);
assertZero[name_, expr_, assumptions_: True] := Module[{r},
  r = FullSimplify[expr, Assumptions -> assumptions];
  record[name, And @@ (# === 0 & /@ Flatten[{r}]), ToString[r, InputForm]];
];
emit[name_, expr_] := (
  AssociateTo[expressions, name -> ToString[expr, InputForm]];
  Print[name, " = ", ToString[expr, InputForm]];
);
real = Element[{theta, phi, bp, bl, omegaB, t, phase0, psi, x, charge, light, radius}, Reals];
physical = real && 0 < theta < Pi && bp > 0 && bp^2 + bl^2 < 1 &&
  omegaB > 0 && charge > 0 && light > 0 && radius > 0;

(* 1. Geometry and the orbit, solved from the Lorentz ODE. *)
n = {0, 0, 1};
bhat = {Sin[theta] Cos[phi], Sin[theta] Sin[phi], Cos[theta]};
etheta = D[bhat, theta]; ephi = Cross[bhat, etheta];
localBasis = {etheta, ephi, bhat};
eparallel = FullSimplify[(bhat - (bhat.n) n)/Sin[theta], physical];
eperp = Cross[n, eparallel];
assertZero["local basis orthonormal", localBasis.Transpose[localBasis] - IdentityMatrix[3], real];
assertZero["local basis right handed", Cross[etheta, ephi] - bhat, real];
assertZero["sky basis right handed", Cross[eparallel, eperp] - n, real];
assertZero["sky perpendicular equals e_phi", eperp - ephi, real];
localVelocity = DSolveValue[{
  u'[t] == -omegaB v[t], v'[t] == omegaB u[t], w'[t] == 0,
  u[0] == bp Cos[phase0], v[0] == bp Sin[phase0], w[0] == bl
}, {u[t], v[t], w[t]}, t];
velocity = FullSimplify[localVelocity.localBasis, physical];
acceleration = D[velocity, t];
assertZero["Lorentz equation", acceleration + omegaB Cross[velocity, bhat], physical];
assertZero["constant speed", velocity.velocity - bp^2 - bl^2, physical];
assertZero["no magnetic work", velocity.acceleration, physical];
position = FullSimplify[Integrate[light velocity, t], physical];
assertZero["trajectory derivative", D[position, t] - light velocity, physical];
dop = 1 - bl Cos[theta];
arrival = FullSimplify[t - n.position/light, physical];
phase = omegaB t + phase0;
assertZero["retarded arrival time", arrival - dop t - bp Sin[theta] Sin[phase]/omegaB, physical];
assertZero["arrival-time Jacobian", D[arrival, t] - (1 - n.velocity), physical];
assertZero["observed orbit period", (arrival /. t -> t + 2 Pi/omegaB) - arrival - dop 2 Pi/omegaB, physical];
lineFrequency = FullSimplify[2 Pi/(dop 2 Pi/omegaB) m];
argument = m bp Sin[theta]/dop;
assertZero["harmonic phase", lineFrequency arrival - (m (phase - phase0) + argument Sin[phase]), physical];
emit["arrivalTime", arrival]; emit["lineFrequency", lineFrequency];

(* 2. Radiation and integration by parts, first for an arbitrary trajectory. *)
vGeneral = {vx[t], vy[t], vz[t]};
jacGeneral = 1 - n.vGeneral;
projectGeneral = Cross[n, Cross[n, vGeneral]];
radiationGeneral = Cross[n, Cross[n - vGeneral, D[vGeneral, t]]];
assertZero["general radiation integration-by-parts identity",
  D[projectGeneral/jacGeneral, t] - radiationGeneral/jacGeneral^2];
(* dt=jac dt' cancels one power; the phase derivative i omega jac
   cancels the last one. Boundary terms vanish for a whole orbit at a line. *)
jacobian = FullSimplify[1 - n.velocity, physical];
electric = -charge/(light radius) Cross[n, Cross[n - velocity, acceleration]]/jacobian^3;
fieldSky = FullSimplify[{electric.eparallel, electric.eperp}, physical];
expectedField = charge omegaB bp/(light radius jacobian^3) {
  -(Cos[theta] - bl) Sin[phase], dop Cos[phase] + bp Sin[theta]};
assertZero["time-domain field components", fieldSky - expectedField, physical];
projected = FullSimplify[{Cross[n, Cross[n, velocity]].eparallel,
  Cross[n, Cross[n, velocity]].eperp}, physical];
assertZero["projected velocity", projected - {
  -bl Sin[theta] - bp Cos[theta] Cos[phase], -bp Sin[phase]}, physical];
emit["projectedVelocity", projected]; emit["electricField", fieldSky];

(* 3. Derive the Fourier coefficient from the exponential power series.
   exp(i x sin psi)=exp[(x/2)e^(i psi)] exp[-(x/2)e^(-i psi)].
   For the exp(+i m psi) integral, orthogonality selects s=r+m.
   Both exponential series converge absolutely for finite real x. *)
record["Fourier zero mode", Integrate[1, {psi, 0, 2 Pi}]/(2 Pi) === 1];
assertZero["Fourier nonzero integer modes", Integrate[Exp[I k psi], {psi, 0, 2 Pi},
  Assumptions -> Element[k, Integers] && k != 0], Element[k, Integers] && k != 0];
seriesAssumptions = Element[m, Integers] && m >= 1 && x > 0;
unsignedCoefficient = FullSimplify[Sum[(-1)^r (x/2)^(m+2 r)/(r! (m+r)!),
  {r, 0, Infinity}], seriesAssumptions];
assertZero["Bessel coefficient from exponential series", unsignedCoefficient - BesselJ[m, x], seriesAssumptions];
f0 = (-1)^m unsignedCoefficient;
(* The cosine moment follows from integrating d(exp(i(m psi+x sin psi)))/dpsi.
   The sine moment follows by differentiating the convergent coefficient in x. *)
fcos = -m f0/x; fsin = D[f0, x]/I;
amplitudeDerived = FullSimplify[{
  -bl Sin[theta] f0 - bp Cos[theta] fcos, -bp fsin} /. x -> argument,
  physical && Element[m, Integers] && m >= 1];
amplitudeTarget = (-1)^m {
  (Cos[theta] - bl)/Sin[theta] BesselJ[m, argument],
  I bp Derivative[0, 1][BesselJ][m, argument]};
assertZero["both complex harmonic amplitudes", amplitudeDerived - amplitudeTarget,
  physical && Element[m, Integers] && m >= 1];
emit["harmonicAmplitudes", amplitudeDerived];
(* Exact recurrence used by the Python implementation at the viewing axis. *)
assertZero["axis-safe amplitude recurrence",
  BesselJ[m - 1, x] + BesselJ[m + 1, x] - 2 m BesselJ[m, x]/x, seriesAssumptions];

(* 4. Absolute source-time power from Gaussian-cgs Poynting flux and Parseval.
   W/dOmega=c R^2/(4pi) int E(t)^2 dt. For a real field the +/-omega
   halves of Parseval coincide: positive-frequency coefficient c R^2/(4pi^2).
   Integration by parts contributes e^2 omega^2/(c^2 R^2). *)
positiveEnergyFactor = light radius^2/(4 Pi) * 2/(2 Pi);
velocityEnergyFactor = FullSimplify[positiveEnergyFactor charge^2 freq^2/(light^2 radius^2)];
(* With z=T frequency_offset, T sinc^2(T frequency_offset/2) has mass 2pi.
   It is a nonnegative approximate identity; this is a weak, not pointwise, limit. *)
lineMass = Integrate[(2 Sin[z/2]/z)^2, {z, -Infinity, Infinity}];
assertZero["finite-time line mass", lineMass - 2 Pi];
tailEnvelope = Integrate[8/(duration z^2), {z, epsilon, Infinity},
  Assumptions -> duration > 0 && epsilon > 0];
assertZero["line-kernel tail vanishes", tailEnvelope - 8/(duration epsilon), duration > 0 && epsilon > 0];
lineIntegral = Integrate[freq^2 DiracDelta[d freq - a], {freq, -Infinity, Infinity},
  Assumptions -> d > 0 && a > 0];
assertZero["delta Jacobian with frequency-squared weight", lineIntegral - a^2/d^3, d > 0 && a > 0];
prefactor = FullSimplify[(velocityEnergyFactor/freq^2) lineMass (lineIntegral /. {d -> dop, a -> m omegaB})];
assertZero["absolute harmonic power prefactor", prefactor - charge^2 m^2 omegaB^2/(2 Pi light dop^3)];
emit["sourceTimePowerPrefactor", prefactor];

(* 5. Stokes from the coherency matrix in the ordered (parallel,perp) basis. *)
stokes[jones_] := Module[{coherency = Outer[Times, jones, Conjugate[jones]]},
  {Tr[coherency], coherency[[1, 1]] - coherency[[2, 2]],
   2 Re[coherency[[1, 2]]], -2 Im[coherency[[1, 2]]]}
];
sNatural = FullSimplify[stokes[{aa, I bb}], Element[{aa, bb}, Reals]];
assertZero["natural-basis I Q U V", sNatural - {aa^2+bb^2, aa^2-bb^2, 0, 2 aa bb}, Element[{aa, bb}, Reals]];
assertZero["complete single-harmonic polarisation", sNatural[[1]]^2 - Total[sNatural[[2;;4]]^2], Element[{aa, bb}, Reals]];
rotation = Transpose[{eparallel[[1;;2]], eperp[[1;;2]]}];
jonesGeneral = {ar + I ai, br + I bi};
stokesGeneral = ComplexExpand[stokes[jonesGeneral]];
stokesSky = FullSimplify[ComplexExpand[stokes[rotation.jonesGeneral]], Element[phi, Reals]];
rotationTarget = {stokesGeneral[[1]],
  stokesGeneral[[2]] Cos[2 phi] - stokesGeneral[[3]] Sin[2 phi],
  stokesGeneral[[2]] Sin[2 phi] + stokesGeneral[[3]] Cos[2 phi], stokesGeneral[[4]]};
assertZero["general Jones-to-sky Stokes rotation", stokesSky - rotationTarget, Element[{phi, ar, ai, br, bi}, Reals]];
record["positive-V circular state", stokes[{1, I}/Sqrt[2]] === {1, 0, 0, 1}];
(* The same real field has conjugate phasors if the time convention is reversed. *)
assertZero["V unchanged under consistent Fourier conjugation",
  -2 Im[jonesGeneral[[1]] Conjugate[jonesGeneral[[2]]]] -
   2 Im[Conjugate[jonesGeneral[[1]]] jonesGeneral[[2]]], Element[{ar, ai, br, bi}, Reals]];
emit["naturalStokes", sNatural];
emit["skyStokes", FullSimplify[ComplexExpand[stokes[rotation.{aa, I bb}]], Element[{aa, bb, phi}, Reals]]];

(* 6. Independent total power: integrate the retarded field's source-time flux.
   Rotate the instantaneous velocity to z and its perpendicular acceleration to x.
   This uses the same physical field, not a sum normalised to a target total. *)
sightline = {Sqrt[1 - mu^2] Cos[az], Sqrt[1 - mu^2] Sin[az], mu};
num = Cross[sightline, Cross[sightline - {0, 0, speed}, {acc, 0, 0}]];
numSquared = FullSimplify[num.num, Element[{mu, az, speed, acc}, Reals] && -1 < mu < 1];
azimuthIntegral = Integrate[numSquared, {az, 0, 2 Pi}];
angularIntegral = Integrate[azimuthIntegral/(1 - speed mu)^5, {mu, -1, 1},
  Assumptions -> 0 < speed < 1 && acc > 0];
powerFromField = FullSimplify[charge^2/(4 Pi light) angularIntegral];
assertZero["total power from angular Poynting integral",
  powerFromField - 2 charge^2 acc^2/(3 light (1 - speed^2)^2),
  charge > 0 && light > 0 && acc > 0 && 0 < speed < 1];
emit["totalPower", powerFromField];
helicalTotal = FullSimplify[powerFromField /. {acc -> omegaB bp, speed -> Sqrt[1 - 1/lorentz^2]}];
assertZero["helical total power", helicalTotal - 2 charge^2 lorentz^4 omegaB^2 bp^2/(3 light)];
coolingRatio = FullSimplify[(lorentz mass light^2/helicalTotal)/(2 Pi/omegaB) /.
  omegaB -> charge field/(lorentz mass light)];
assertZero["cooling time over gyroperiod", coolingRatio - 3 mass^2 light^4/(4 Pi charge^3 field lorentz^2 bp^2)];
emit["coolingOrbitRatio", coolingRatio];

numericRun = Check[Get[FileNameJoin[{root, "numerical_checks.wls"}]], $Failed];
record["numerical checks completed without Wolfram messages", numericRun =!= $Failed];
report = <|"wolframVersion" -> $Version, "dateUTC" -> DateString[Now, "ISODateTime", TimeZone -> 0],
  "scope" -> "Classical prescribed helical orbit; vacuum far zone; source-time line powers",
  "scriptsSHA256" -> AssociationMap[FileHash[FileNameJoin[{root, #}], "SHA256", "HexString"] &,
    {"verify.wls", "numerical_checks.wls"}],
  "checks" -> checks, "expressions" -> expressions, "numericalCases" -> numericalCases,
  "allPassed" -> And @@ Lookup[checks, "pass"]|>;
Export[FileNameJoin[{root, "verification.json"}], report, "RawJSON"];
Print[Count[Lookup[checks, "pass"], True], "/", Length[checks], " checks passed."];
Exit[If[TrueQ[report["allPassed"]], 0, 1]];
