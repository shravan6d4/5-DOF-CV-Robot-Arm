%% =========================================================================
%  test_ik_fk.m  --  offline regression for the IK/FK server logic
%
%  Runs the 8 named targets from IKtrials_v2.m's regression battery (Part 6)
%  through the SAME init + IK path the TCP server uses (init_arm.m), with no
%  tcpserver and no Simulink. This is the Stage-1 sanity gate: if these errors
%  match IKtrials_v2 (all well under the 10 mm tolerance), the refactor into
%  the server preserved the validated solver.
%
%  For each target it also re-derives the wrist/tip config the way
%  handle_fk_request does (rebuild from homeAngles + J1..J5 overrides) and
%  confirms that reproduces the solved tip position — i.e. the IK->angles->FK
%  round-trip the Python client relies on is self-consistent.
%
%  Run from MATLAB:  >> test_ik_fk        (or press Run / F5)
% =========================================================================

clear; clc;
global robot ik motorIdx homeAngles endEffector

init_arm;

names = {'fwd-mid','fwd-high','left','right','diag','near','low','stretch'};
tgts = [0.150 0 -0.080; 0.150 0 0; 0 0.180 -0.060; 0 -0.180 -0.060; ...
        0.120 0.120 -0.100; 0.080 0.060 -0.040; 0.140 -0.060 -0.160; ...
        0.220 0 -0.060];

weights = [0 0 0 1 1 1];   % position only
fprintf('\n%-9s %16s %10s %14s\n', 'target', 'xyz [mm]', 'IK err', 'FK roundtrip');
fprintf('%s\n', repmat('-', 1, 54));

worstErr = 0; worstRt = 0;
for i = 1:8
    p = tgts(i,:);

    % --- IK: multi-restart, same as handle_ik_request ---
    seed = homeConfiguration(robot); bestErr = inf; bestSol = [];
    for a = 1:10
        [s, ~] = ik(endEffector, trvec2tform(p), weights, seed);
        T = getTransform(robot, s, endEffector);
        e = norm(T(1:3,4) - p');
        if e < bestErr, bestErr = e; bestSol = s; end
        if bestErr < 0.002, break; end
        seed = randomConfiguration(robot);
    end

    % --- Extract J1..J5, then rebuild config the way FK does ---
    ik_angles = homeAngles;
    for k = 1:5, ik_angles(k) = bestSol(motorIdx(k)); end
    cfg = homeConfiguration(robot);
    for k = 1:5, cfg(motorIdx(k)) = ik_angles(k); end
    Ttip = getTransform(robot, cfg, endEffector);
    rt = norm(Ttip(1:3,4) - p');   % should equal bestErr if extraction is faithful

    worstErr = max(worstErr, bestErr);
    worstRt  = max(worstRt, rt);
    fprintf('%-9s [%4.0f %4.0f %4.0f] %8.2f mm %10.2f mm\n', ...
            names{i}, 1000*p, 1000*bestErr, 1000*rt);
end

fprintf('%s\n', repmat('-', 1, 54));
fprintf('worst IK err = %.2f mm   worst roundtrip = %.2f mm\n', 1000*worstErr, 1000*worstRt);
if worstErr < 0.010
    fprintf('PASS: all 8 targets within 10 mm tolerance.\n');
else
    fprintf(2, 'FAIL: a target exceeded the 10 mm tolerance.\n');
end
