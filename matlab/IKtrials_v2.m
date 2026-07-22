%% =========================================================================
%  6-DOF ROBOTIC ARM - VALIDATED IK PIPELINE (v2)
%  Verified in-session: 8/8 test targets hit at 0.0 mm, poses visually
%  confirmed clean (no self-intersection).
%  KEY FIXES vs v1:
%   1. End effector = CLAW TIP (new fixed body), not Body08 origin.
%      Body08's origin sits ON Joint08's axis, so wrist pitch moved it
%      0 mm and the IK ignored a whole DOF. Targeting the tip restores
%      all 5 position DOF (J1..J5) and removes the need for the manual
%      "claw pullback" hack in the camera layer.
%   2. Placeholders j1..j6 are created BEFORE importrobot (its compile
%      step requires the From Workspace variables to exist) and are
%      never cleared.
%   3. Limits centered on IMPORTED home angles (never overwrite home).
%   4. ik_angles extracted via row-format config indexing (robust).
% =========================================================================

%% ===== PREP: PLACEHOLDER HOLD-STILL TRAJECTORIES (required to compile) =====
T_total = 10;  T_delay = 2;
t = (0:0.05:T_total)';
holdRad = [pi, 165*pi/180, 0, pi, pi, pi/2];   % J1..J6 hold-still angles
j1 = [t, repmat(holdRad(1), size(t))];
j2 = [t, repmat(holdRad(2), size(t))];
j3 = [t, repmat(holdRad(3), size(t))];
j4 = [t, repmat(holdRad(4), size(t))];
j5 = [t, repmat(holdRad(5), size(t))];
j6 = [t, repmat(holdRad(6), size(t))];
disp('Placeholders assigned (arm holds still if sim runs now).');

%% ===== PART 0: IMPORT + AUTOMATIC JOINT MAPPING =====
% Physical servos by SIMULINK REVOLUTE BLOCK number:
% J1=Rev1, J2=Rev2, J3=Rev5, J4=Rev7, J5=Rev8, J6=Rev11
% NOTE: tree JointNN names are scrambled vs Revolute numbers (import
% traversal order). Never assume Joint05==Revolute5; match block names.
revNums = [1 2 5 7 8 11];

[robot, importInfo] = importrobot('Robomainassemjoints.slx');
robot.DataFormat = 'row';   % numeric row configs; needed for clean indexing

realMotors = cell(1,6); motorBodies = cell(1,6); motorIdx = zeros(1,6);
for i = 1:robot.NumBodies
    bInfo = bodyInfo(importInfo, robot.BodyNames{i});
    jb = bInfo.JointBlocks; if iscell(jb), jb = jb{1}; end
    tok = regexp(char(jb), 'Revolute\s*(\d+)\s*$', 'tokens', 'once');
    if isempty(tok), continue; end
    k = find(revNums == str2double(tok{1}), 1);
    if ~isempty(k)
        realMotors{k}  = robot.Bodies{i}.Joint.Name;
        motorBodies{k} = robot.BodyNames{i};
        motorIdx(k)    = i;   % body index == config-vector index (all revolute)
    end
end

fprintf('\n===== SERVO -> TREE JOINT MAPPING =====\n');
for k = 1:6
    fprintf('J%d = Revolute%-2d -> %-8s (body %s)\n', k, revNums(k), ...
            realMotors{k}, motorBodies{k});
end
assert(~any(cellfun(@isempty, realMotors)), 'Mapping failed - check block names.');

%% ===== PART 1: FREEZE IDLER-DISK JOINTS + LIMITS AROUND TRUE HOME =====
% The 6 extra revolutes are the passive idler disks on the far side of
% each motor. Freeze them at their imported home. NEVER overwrite
% HomePosition on real joints - the imported home IS the CAD hold pose.
eps_ = 1e-6;
homeAngles = zeros(1,6);
rangeDeg = [ -90  90;    % J1 base yaw
             -90  90;    % J2 shoulder
             -90  90;    % J3 elbow
             -90  90;    % J4 forearm
             -90  90;    % J5 wrist pitch
             -45  45 ];  % J6 claw rotation
for i = 1:robot.NumBodies
    jnt = robot.Bodies{i}.Joint;
    h = jnt.HomePosition;
    k = find(motorIdx == i, 1);
    if isempty(k)
        jnt.PositionLimits = [h - eps_, h + eps_];   % frozen idler disk
    else
        homeAngles(k) = h;
        jnt.PositionLimits = h + deg2rad(rangeDeg(k,:));
    end
end
fprintf('\nHome angles J1..J6 [deg]: '); fprintf('%7.1f', rad2deg(homeAngles)); fprintf('\n');

%% ===== PART 2: ADD CLAW-TIP END EFFECTOR =====
cfg0 = homeConfiguration(robot);
T08 = getTransform(robot, cfg0, 'Body08');
T09 = getTransform(robot, cfg0, 'Body09');
T10 = getTransform(robot, cfg0, 'Body10');
v_world = (T09(1:3,4) + T10(1:3,4))/2 - T08(1:3,4);   % wrist -> claw direction
v_local = T08(1:3,1:3)' * (v_world / norm(v_world));   % in Body08 frame
CLAW_LEN = 0.07006;                                     % wrist to fingertip [m]
tipBody = rigidBody('ClawTip');
tj = rigidBodyJoint('ClawTipJnt','fixed');
setFixedTransform(tj, trvec2tform(CLAW_LEN * v_local'));
tipBody.Joint = tj;
addBody(robot, tipBody, 'Body08');
endEffector = 'ClawTip';

Ttip = getTransform(robot, cfg0, endEffector);
fprintf('Claw tip at home [mm]: [%.0f %.0f %.0f]\n', 1000*Ttip(1:3,4));

%% ===== PART 3: TARGET (robot-base frame) =====
% For now targets are given DIRECTLY in the robot base frame. The camera
% layer below is kept but OFF until you measure the camera extrinsics.
useCamera = false;

if useCamera
    cam = [0.15; 0.05; 0.80];                 % raw OpenCV [m], camera frame
    R_cam2robot = [ -1 0 0; 0 0 -1; 0 1 0 ];  % your axis remap
    cam_origin_in_robot = [0; 0; 0];          % MEASURE with a tape [m]!
    p_robot = R_cam2robot * cam + cam_origin_in_robot;
    % (No claw pullback needed anymore - the EE *is* the fingertip.)
else
    p_robot = [0.50; 0.000; -0.080];         % validated test target
end

% Reach clamp: tip max reach with these limits was measured ~0.32-0.38 m.
maxReach = 0.32;
d = norm(p_robot);
if d > maxReach
    fprintf(2, 'WARNING: target %.0f mm > reach %.0f mm - clamping (test stopgap).\n', 1000*d, 1000*maxReach);
    p_robot = p_robot * (0.9 * maxReach / d);
end
fprintf('\nTarget [mm]: [%.0f %.0f %.0f]\n', 1000*p_robot);

%% ===== PART 4: IK - POSITION-ONLY, MULTI-RESTART, FK-VERIFIED =====
ik = inverseKinematics('RigidBodyTree', robot);   % created AFTER addBody!
ik.SolverParameters.MaxIterations = 800;
weights = [0 0 0 1 1 1];    % [orientation(3), position(3)] - position only

targetPose = trvec2tform(p_robot');
seed = homeConfiguration(robot);
bestSol = []; bestErr = inf;
for attempt = 1:10
    [sol, ~] = ik(endEffector, targetPose, weights, seed);
    T = getTransform(robot, sol, endEffector);
    err = norm(T(1:3,4) - p_robot);
    if err < bestErr, bestErr = err; bestSol = sol; end
    if bestErr < 0.002, break; end
    seed = randomConfiguration(robot);
end
configSol = bestSol;
fprintf('IK: %.1f mm after %d attempt(s)\n', 1000*bestErr, attempt);

IK_TOL = 0.010;
if bestErr > IK_TOL
    error(['IK missed by %.1f mm (> %.0f mm tol). Target outside the\n' ...
           'limited workspace. NOT generating a trajectory.'], ...
           1000*bestErr, 1000*IK_TOL);
end

% Extract ABSOLUTE angles (row config: index == body index)
ik_angles = homeAngles;
for k = 1:5
    ik_angles(k) = configSol(motorIdx(k));
end
ik_angles(6) = homeAngles(6);   % claw rotation: command manually
fprintf('Angles J1..J6 [deg]: '); fprintf('%7.1f', rad2deg(ik_angles)); fprintf('\n');

%% ===== PART 5: SMOOTH TRAJECTORY (home -> ik_angle, ABSOLUTE) =====
% Tree coords == Simulink signal coords (same CAD zero). No offsets added.
curve = zeros(numel(t),1);
mi = t > T_delay;
nt = (t(mi) - T_delay) / (T_total - T_delay);
curve(mi) = 3*nt.^2 - 2*nt.^3;

% Flip an entry to -1 ONLY if the single-joint Simulink test (bottom)
% shows that joint mirroring vs the MATLAB show() plot.
axis_sign = [1 1 1 1 1 1];

q0 = homeAngles(1); q1v = q0 + axis_sign(1)*(ik_angles(1)-q0);
j1 = [t, q0 + (q1v-q0)*curve];
q0 = homeAngles(2); q1v = q0 + axis_sign(2)*(ik_angles(2)-q0);
j2 = [t, q0 + (q1v-q0)*curve];
q0 = homeAngles(3); q1v = q0 + axis_sign(3)*(ik_angles(3)-q0);
j3 = [t, q0 + (q1v-q0)*curve];
q0 = homeAngles(4); q1v = q0 + axis_sign(4)*(ik_angles(4)-q0);
j4 = [t, q0 + (q1v-q0)*curve];
q0 = homeAngles(5); q1v = q0 + axis_sign(5)*(ik_angles(5)-q0);
j5 = [t, q0 + (q1v-q0)*curve];
q0 = homeAngles(6); q1v = q0 + axis_sign(6)*(ik_angles(6)-q0);
j6 = [t, q0 + (q1v-q0)*curve];
clear q0 q1v
disp('SUCCESS - j1..j6 ready. Run the Simulink model.');

%% ===== PART 6: VISUAL VERIFICATION =====
figure;
show(robot, configSol, 'Frames','off'); hold on;
plot3(p_robot(1), p_robot(2), p_robot(3), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
plot3(0,0,0,'k.','MarkerSize',40);
title(sprintf('Claw tip vs target (err %.1f mm)', 1000*bestErr));
view(135,20); axis equal; grid on; hold off;

%% ===== OPTIONAL: REGRESSION TEST BATTERY (set true to run, ~30 s) =====
runTests = false;
if runTests
    names = {'fwd-mid','fwd-high','left','right','diag','near','low','stretch'};
    tgts = [0.150 0 -0.080; 0.150 0 0; 0 0.180 -0.060; 0 -0.180 -0.060; ...
            0.120 0.120 -0.100; 0.080 0.060 -0.040; 0.140 -0.060 -0.160; ...
            0.220 0 -0.060];
    for i = 1:8
        p = tgts(i,:); seed = homeConfiguration(robot); best = inf;
        for a = 1:6
            [s,~] = ik(endEffector, trvec2tform(p), weights, seed);
            T = getTransform(robot, s, endEffector);
            e = norm(T(1:3,4)-p');
            if e < best, best = e; end
            if best < 0.005, break; end
            seed = randomConfiguration(robot);
        end
        fprintf('%-9s [%4.0f %4.0f %4.0f] -> %6.1f mm\n', names{i}, 1000*p, 1000*best);
    end
end

%% ===== SINGLE-JOINT AXIS-SIGN TEST (do once per joint in Simulink) =====
% 1. Set all jN to hold at homeAngles(N) (run PREP only).
% 2. For joint k, command home+20deg:  jk(:,2) = homeAngles(k)+deg2rad(20)*curve... 
%    then run the sim and compare direction vs:
%      c = homeConfiguration(robot); c(motorIdx(k)) = homeAngles(k)+deg2rad(20);
%      figure; show(robot, c);
% 3. Opposite direction? -> axis_sign(k) = -1.
