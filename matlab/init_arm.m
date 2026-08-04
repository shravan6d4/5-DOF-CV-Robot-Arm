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

%% ===== PART 1: FREEZE IDLER-DISK JOINTS + LIMITS AROUND THE ARM'S REAL ZERO =====
%
% The motor joints' limits are centred on ANGLE ZERO, not on the imported
% model's HomePosition, and their HomePosition is reset to 0 to match.
%
% Why (found 2026-07-22, on hardware): the physical arm operates around MATLAB
% angle ~0 — verified by FK, which at [0 0 0 0 0] puts the claw tip ~70mm in
% front of the base hanging below the wrist, matching the real arm. But the
% imported model's HomePosition sits near [180, 165, 0, 170, 176] deg, so the
% old `h + rangeDeg` centred each +/-90deg band on THAT pose. The arm's actual
% working position then fell OUTSIDE its own joint limits on J1/J2/J4/J5, and
% the IK solver — correctly obeying those limits — could never return a
% solution near where the arm really was. It returned the nearest legal
% posture instead, ~180deg away: a 45mm lift came back wanting ~2600 ticks
% (~227deg) of base rotation. ServoBus's move cap refused them, which is the
% only reason this surfaced as a refusal rather than a wild swing.
%
% Resetting HomePosition (a default configuration value, NOT geometry — the
% link transforms are untouched, so FK at any given angle is unchanged) keeps
% homeConfiguration inside the new limits, so it remains a valid IK seed.
eps_ = 1e-6;
homeAngles = zeros(1,6);
rangeDeg = [ -90  90;    % J1 base yaw
             -90  90;    % J2 shoulder
             -90  90;    % J3 elbow
             -90  90;    % J4 forearm
             -90  90;    % J5 wrist pitch
             -45  45 ];  % J6 claw rotation

% MEASURED limits override the blanket +-90 above, per joint, where they exist.
% Those defaults are a placeholder, not a measurement: the IK solver will
% happily return a solution the arm physically cannot reach, and the servo bus
% then either refuses it or -- before travel limits existed -- drove the joint
% into a hard stop (J3 and J4 both jammed that way on 2026-08-04).
%
% Written by scripts/find_joint_limits.py, which measures each end by small
% operator-confirmed steps. The file is per-arm and gitignored, so a missing
% file simply leaves the defaults in place. Radians, model convention.
limitsFile = fullfile('..','data','joint_limits_rad.json');
if isfile(limitsFile)
    lim = jsondecode(fileread(limitsFile));
    for k = 1:6
        f = sprintf('x%d', k);          % jsondecode prefixes numeric keys
        if isfield(lim, f)
            rangeDeg(k,:) = rad2deg([lim.(f).min_rad, lim.(f).max_rad]);
            fprintf('init_arm: J%d limits from file: [%.1f %.1f] deg\n', ...
                    k, rangeDeg(k,1), rangeDeg(k,2));
        end
    end
else
    fprintf(['init_arm: no %s — using placeholder +-90 deg limits. ' ...
             'Measure with scripts/find_joint_limits.py.\n'], limitsFile);
end
for i = 1:robot.NumBodies
    jnt = robot.Bodies{i}.Joint;
    h = jnt.HomePosition;
    k = find(motorIdx == i, 1);
    if isempty(k)
        jnt.PositionLimits = [h - eps_, h + eps_];   % frozen idler disk (unchanged)
    else
        % Motor joint: re-zero to the arm's real operating point.
        jnt.PositionLimits = deg2rad(rangeDeg(k,:));
        jnt.HomePosition   = 0;
        homeAngles(k)      = 0;
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
