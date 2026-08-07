%% test_arm_from_survey.m -- does the survey-built tree agree with the arm?
%
%   >> test_arm_from_survey
%
% Offline: no server, no serial port, no arm. It reads
% data/servo_calibration.json for the tick->radian conversion and checks the
% tree from build_arm_from_survey.m against measurements it was NOT built from.
%
% THIS IS THE FILE THAT EARNS THE TREE ITS TRUST, and it is deliberately the
% MATLAB twin of tests/test_arm_model.py. The two implementations must agree; if
% MATLAB reports a spread near 6.0 mm the port is faithful, and if it reports 73
% something is wrong HERE rather than in the geometry, because Python already
% gets 6.0 from the same numbers.
%
% Section 3 is the one to read first. The others are arithmetic.

clear; clc;
cd(fileparts(mfilename('fullpath')));

[robot, motorIdx, endEffector, wristBody] = build_arm_from_survey(); %#ok<ASGLU>
fprintf('built: %d bodies, EE %s, wrist %s\n\n', robot.NumBodies, endEffector, wristBody);

%% ---- calibration -----------------------------------------------------------
calFile = fullfile('..','data','servo_calibration.json');
assert(isfile(calFile), 'no %s -- run from matlab/ with the repo intact', calFile);
cal = jsondecode(fileread(calFile));

%% ---- 1. home reproduces the survey -----------------------------------------
fprintf('1. HOME must reproduce the survey (it is what built the tree)\n');
homeTicks = zeros(1,5);
for k = 1:5
    homeTicks(k) = cal.(sprintf('x%d', k)).home_tick;
end
cfg0 = homeConfiguration(robot);
T0 = getTransform(robot, cfg0, endEffector);
fprintf('   tip at home: forward %.2f mm, height %.2f mm   (survey: 75.0, 0.0)\n', ...
        1000*T0(1,4), 1000*T0(3,4));
assert(abs(1000*T0(1,4) - 75.0) < 0.5 && abs(1000*T0(3,4)) < 0.5, ...
       'home does not reproduce the survey -- the chain is assembled wrong');
fprintf('   OK\n\n');

%% ---- 2. link lengths against the directly measured ones --------------------
fprintf('2. LINK LENGTHS -- survey-implied vs measured along the link\n');
want = [102.7, 136.2, NaN, 70.1];
lbl  = {'shoulder->elbow','elbow->wrist pitch','wrist pitch->wrist','wrist->tip'};
S = [12.0 90.0; -72.0 146.0; 67.0 152.0; 74.0 71.3; 75.0 0.0];
for k = 1:4
    L = norm(S(k+1,:) - S(k,:));
    if isnan(want(k))
        fprintf('   %-20s %7.2f mm   (not measured directly)\n', lbl{k}, L);
    else
        fprintf('   %-20s %7.2f mm   ruler %5.1f   diff %+5.2f\n', ...
                lbl{k}, L, want(k), L - want(k));
        assert(abs(L - want(k)) < 3.5, 'link %d disagrees with the ruler', k);
    end
end
fprintf('   OK -- two independent measurement routes agree\n\n');

%% ---- 3. HELD-OUT: seven touches of one flat table ---------------------------
fprintf('3. HELD-OUT TEST -- seven touches of the tabletop\n');
fprintf('   Recorded by scripts/measure_table_plane.py BEFORE the survey existed\n');
fprintf('   and used nowhere in building this tree. All seven are the same flat\n');
fprintf('   plane, so all seven must return the same height. That is a property\n');
fprintf('   of the KINEMATICS, not of the table.\n\n');
touch = [1995 3602 2590 1499 2745;
         2359 3759 2344 1500 2744;
         1883 3759 2340 1567 2746;
         2071 3759 2343 1567 2743;
         2073 3295 2906 1462 2746;
         2321 3295 2912 1464 2745;
         1794 3294 2907 1462 2747];
h = zeros(1, size(touch,1));
for i = 1:size(touch,1)
    h(i) = tipHeight(robot, motorIdx, endEffector, cal, touch(i,:));
    fprintf('   touch %d: %+7.2f mm\n', i, h(i));
end
spread = max(h) - min(h);
fprintf('\n   spread %.1f mm   mean %+.1f mm\n', spread, mean(h));
fprintf('   for reference: imported CAD model 73.1 mm about +20.6\n');
fprintf('                  arm_model.py        6.0 mm about  -1.4\n\n');

if spread > 10.0
    fprintf(2, ['   *** FAILED. Python gets 6.0 mm from these same numbers, so a\n' ...
                '       large spread here means THIS PORT is wrong, not the\n' ...
                '       geometry. Check PITCH_AXIS first: [0 -1 0] makes a positive\n' ...
                '       angle carry +X toward +Z, matching arm_model.py. Flipping it\n' ...
                '       inverts every pitch joint.\n']);
    error('held-out touches spread %.1f mm', spread);
end
assert(abs(mean(h)) < 5.0, 'flat but not at the table: mean %+.1f mm', mean(h));
fprintf('   OK -- matches the Python model, on data neither has seen\n\n');

%% ---- 4. base yaw cannot change the tip height ------------------------------
fprintf('4. J1 is a yaw about vertical, so it must not change height\n');
cfgY = homeConfiguration(robot);
zs = zeros(1,3); d = deg2rad([-40 0 40]);
for i = 1:3
    cfgY(motorIdx(1)) = d(i);
    Ty = getTransform(robot, cfgY, endEffector);
    zs(i) = 1000*Ty(3,4);
end
fprintf('   heights at yaw -40/0/+40 deg: %+.4f %+.4f %+.4f mm\n', zs);
assert(max(zs) - min(zs) < 1e-6, 'yaw changed the tip height -- axis is wrong');
fprintf('   OK\n\n');

fprintf('ALL CHECKS PASSED. This tree is NOT wired in: init_arm.m still imports\n');
fprintf('the CAD model. Switching over also means dropping the diag(1,-1,-1,1)\n');
fprintf('flip in MatlabIKClient, because this tree is already in the physical\n');
fprintf('frame -- see build_arm_from_survey.m "FRAME".\n');


%% ---- local functions (MATLAB requires these at the END of a script) --------

function th = ticksToRad(cal, j, ticks)
% Mirrors servo_calibration.ticks_to_rad. Zero at home_tick, by definition.
c  = cal.(sprintf('x%d', j));       % jsondecode prefixes numeric keys with x
th = c.dir_sign * (ticks - c.home_tick) / c.ticks_per_rad;
end

function z = tipHeight(robot, motorIdx, endEffector, cal, ticks)
% Claw-tip height above the TABLE in mm. The tree's origin sits at table level on
% the base yaw axis, so the tip's z IS its height above the table -- no offset,
% which is the point of choosing that origin.
cfg = homeConfiguration(robot);
for k = 1:5
    cfg(motorIdx(k)) = ticksToRad(cal, k, ticks(k));
end
T = getTransform(robot, cfg, endEffector);
z = 1000 * T(3,4);
end
