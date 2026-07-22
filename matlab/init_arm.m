%% =========================================================================
%  init_arm.m  --  shared arm / IK / FK initialization
%
%  SCRIPT (deliberately not a function): it must run in the BASE workspace so
%  the variables importrobot needs at compile time are visible to it —
%    * smiData      : rigid transforms / solids / joint positions the Simscape
%                     model is parameterized against (from the DataFile);
%    * j1..j6       : the model's "From Workspace" placeholder signals. Its
%                     compile step reads them, so they MUST exist before
%                     importrobot or the import fails (see IKtrials_v2.m KEY FIX #2).
%  A function would put these in a private workspace where the model compile
%  can't see them, so this stays a script and both ik_fk_server.m and
%  test_ik_fk.m simply `init_arm;` at their top level.
%
%  Produces these GLOBALS (consumed by the handlers / harness):
%    robot ik motorIdx homeAngles endEffector wristBody maxReach IK_TOL
%
%  Logic is lifted verbatim from IKtrials_v2.m Parts 0-2 (the validated
%  pipeline: 8/8 targets at 0.0 mm) — only the packaging changed.
% =========================================================================

global robot ik motorIdx homeAngles endEffector wristBody maxReach IK_TOL

% Resolve to this file's own folder so the relative model / DataFile names
% below work no matter where the caller was started from.
cd(fileparts(mfilename('fullpath')));

% smiData the model is parameterized against. importrobot evaluates these
% during its compile step, so load them first. (If the model's PreLoadFcn also
% loads them this is a harmless reassignment.)
Robomainassem_DataFile;

%% ===== PREP: placeholder hold-still trajectories (required to compile) =====
% The model's From Workspace blocks reference j1..j6; they must exist BEFORE
% importrobot or the compile step fails (IKtrials_v2.m KEY FIX #2).
T_total = 10;  T_delay = 2;
t = (0:0.05:T_total)';
holdRad = [pi, 165*pi/180, 0, pi, pi, pi/2];   % J1..J6 hold-still angles
j1 = [t, repmat(holdRad(1), size(t))];
j2 = [t, repmat(holdRad(2), size(t))];
j3 = [t, repmat(holdRad(3), size(t))];
j4 = [t, repmat(holdRad(4), size(t))];
j5 = [t, repmat(holdRad(5), size(t))];
j6 = [t, repmat(holdRad(6), size(t))];

%% ===== PART 0: IMPORT + AUTOMATIC JOINT MAPPING =====
revNums = [1 2 5 7 8 11];
[robot, importInfo] = importrobot('Robomainassemjoints.slx');
robot.DataFormat = 'row';

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
        motorIdx(k)    = i;
    end
end
assert(~any(cellfun(@isempty, realMotors)), 'Mapping failed - check block names.');

%% ===== PART 1: FREEZE IDLER-DISK JOINTS + LIMITS AROUND TRUE HOME =====
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

%% ===== PART 2: ADD CLAW-TIP END EFFECTOR =====
cfg0 = homeConfiguration(robot);
T08 = getTransform(robot, cfg0, 'Body08');
T09 = getTransform(robot, cfg0, 'Body09');
T10 = getTransform(robot, cfg0, 'Body10');
v_world = (T09(1:3,4) + T10(1:3,4))/2 - T08(1:3,4);
v_local = T08(1:3,1:3)' * (v_world / norm(v_world));
CLAW_LEN = 0.07006;
tipBody = rigidBody('ClawTip');
tj = rigidBodyJoint('ClawTipJnt','fixed');
setFixedTransform(tj, trvec2tform(CLAW_LEN * v_local'));
tipBody.Joint = tj;
addBody(robot, tipBody, 'Body08');
endEffector = 'ClawTip';
wristBody = 'Body08';

% Config constants for IK
maxReach = 0.32;
IK_TOL = 0.010;

% Create IK solver (must happen AFTER addBody)
ik = inverseKinematics('RigidBodyTree', robot);
ik.SolverParameters.MaxIterations = 800;

fprintf('init_arm: robot ready (%d bodies), ClawTip EE, IK solver built.\n', robot.NumBodies);
