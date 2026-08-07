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

%% ===== GEOMETRY SOURCE =====================================================
%
% TRUE  -> build the arm from the RULER SURVEY (build_arm_from_survey.m)
% FALSE -> the legacy path: importrobot on Robomainassemjoints.slx
%
% Default TRUE since 2026-08-07, because the imported CAD is the wrong shape.
% At the home pose, heights above the tabletop: shoulder model 158.2 / ruler 90,
% elbow 106.4 / 146, wrist pitch 108.0 / 152. The shoulder settles it -- that
% shaft is bolted to the base column, so no joint angle or calibration constant
% can move it, and the CAD reports 158.2 at every pose. The survey tree puts
% seven held-out tabletop touches within a 6.0 mm spread; the CAD spreads them
% over 73.1 mm. See test_arm_from_survey.m, which checks this offline.
%
% This matters most for IK, not FK: a wrong FK gives a bad reading, but a wrong
% IK means every commanded pose is solved in an arm that does not exist. That is
% what drove the claw past the tabletop on 2026-08-07 until J2 stalled 34 ticks
% from its limit.
%
% Flip to FALSE to get the old behaviour back; nothing below this block changed.
USE_SURVEY_GEOMETRY = true;

if USE_SURVEY_GEOMETRY
    [robot, motorIdx, endEffector, wristBody] = build_arm_from_survey();

    % ---- put it in the frame the rest of the project already speaks ---------
    %
    % build_arm_from_survey works in the natural frame: origin ON the base yaw
    % axis at TABLE level, +X the arm's forward, +Z up. Everything above the
    % wire speaks the legacy MODEL frame instead, where MatlabIKClient applies
    % physical = model(x, -y, -z) and heights are measured from the base origin
    % with the table at TABLE_Z_IN_BASE. Converting HERE keeps that contract, so
    % no Python changes and the 26 files that call this server are untouched.
    %
    % Z is matched exactly, which is the axis that matters: it is the one
    % target_z and the floor guard are expressed in. X and Y are NOT rotated
    % onto the CAD's axes -- they now mean "the arm's forward" and "left", which
    % is what they should always have meant. The CAD frame was ~89.4 deg off the
    % arm's own forward (audit_model_axes.py section C), a known unfixed wart, so
    % nothing correct depended on it. The descent is unaffected either way: it
    % builds targets as tip + delta, so the absolute frame cancels, and its reach
    % direction is measured (visual_servo.reach_axis_xy) rather than assumed.
    % J1'S ROTATION SENSE IS UNCHANGED from the legacy tree -- confirmed by the
    % operator on 2026-08-07, who knows this arm's base yaw by eye. It was the
    % one thing this rebuild could have silently inverted: positive J1 swings the
    % tip toward physical +y (LEFT, viewed from behind the arm along the claw;
    % counterclockwise seen from above), and J1's dir_sign +1 was jogged against
    % the OLD tree. Had the sense flipped, that sign would now be backwards, and
    % a wrong sign on the SIDEWAYS axis is what turns the visual servo's
    % correction loop from converging into a runaway -- which is how J1 was run
    % away twice already. It did not flip; no dir_sign needs revisiting.
    TABLE_Z_M = -0.0732;                       % config.TABLE_Z_IN_BASE
    flipX = [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1];
    baseTf = flipX * trvec2tform([0 0 TABLE_Z_M]);
    firstJoint = robot.Bodies{motorIdx(1)}.Joint;
    setFixedTransform(firstJoint, baseTf * firstJoint.JointToParentTransform);

    % ---- joint limits, same file and same meaning as the legacy path --------
    limitsFile = fullfile('..','data','joint_limits_rad.json');
    if isfile(limitsFile)
        lim = jsondecode(fileread(limitsFile));
        for k = 1:5
            f = sprintf('x%d', k);
            if isfield(lim, f)
                robot.Bodies{motorIdx(k)}.Joint.PositionLimits = ...
                    [lim.(f).min_rad, lim.(f).max_rad];
                fprintf('init_arm: J%d limits from file: [%.1f %.1f] deg\n', ...
                        k, rad2deg(lim.(f).min_rad), rad2deg(lim.(f).max_rad));
            else
                fprintf(2, ['init_arm: *** J%d has NO measured limits -- it keeps ' ...
                            'the +-pi default, which is WIDER than several measured\n' ...
                            'init_arm:     ranges, so IK will spend it before joints ' ...
                            'that ARE limited. python scripts/find_joint_limits.py --joint %d\n'], k, k);
            end
        end
    end

    homeAngles = zeros(1,6);
    maxReach   = 0.32;
    IK_TOL     = 0.010;
    ik = inverseKinematics('RigidBodyTree', robot);
    ik.SolverParameters.MaxIterations = 800;

    cfgChk = homeConfiguration(robot);
    Tchk = getTransform(robot, cfgChk, endEffector);
    fprintf(['init_arm: SURVEY geometry. Claw tip at home, model frame: ' ...
             '(%.1f, %.1f, %.1f) mm\n'], 1000*Tchk(1,4), 1000*Tchk(2,4), 1000*Tchk(3,4));
    fprintf(['init_arm:   -> physical z %.1f mm, i.e. %.1f mm above the table. ' ...
             'The ruler says 0.\n'], -1000*Tchk(3,4), -1000*Tchk(3,4) - 1000*TABLE_Z_M);
    fprintf('init_arm: robot ready (%d bodies), ClawTip EE, IK solver built.\n', ...
            robot.NumBodies);
    return;   % skip the legacy import entirely
end

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
% Joint roles below were read back out of the model's own FK by
% scripts/audit_model_axes.py and confirmed against the physical arm on
% 2026-08-06: yaw, three PARALLEL pitches, then a wrist ROLL. J5 used to be
% labelled "wrist pitch" here, which was a mislabel in this comment only --
% the model's geometry always had it as a roll (0.2 deg off the forearm), and
% the operator confirmed the physical joint spins the claw rather than
% tilting it. Link lengths J2->J3 = 102.7 mm, J3->J4 = 136.2 mm.
rangeDeg = [ -90  90;    % J1 base yaw
             -90  90;    % J2 shoulder   pitch
             -90  90;    % J3 elbow      pitch
             -90  90;    % J4 wrist      pitch
             -90  90;    % J5 wrist      ROLL (spins the claw; camera rides here)
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
% A MISSING ENTRY IS NOT A NEUTRAL DEFAULT -- IT IS A WIDE ONE, AND THE SOLVER
% PREFERS IT. The placeholder is +-90 deg, about +-1024 ticks, while a measured
% joint might get a third of that. Faced with a redundant arm the solver spends
% whichever joints look free, so the joints with real limits are spared and the
% unmeasured ones absorb the motion -- the exact opposite of what the missing
% measurement implies. On 2026-08-07 a descent walked J4 into its hard stop over
% four steps while J2 and J3, both properly limited, sat with hundreds of ticks
% of headroom. So say plainly which joints are running on fiction.
limitsFile = fullfile('..','data','joint_limits_rad.json');
measured = false(1,6);
if isfile(limitsFile)
    lim = jsondecode(fileread(limitsFile));
    for k = 1:6
        f = sprintf('x%d', k);          % jsondecode prefixes numeric keys
        if isfield(lim, f)
            rangeDeg(k,:) = rad2deg([lim.(f).min_rad, lim.(f).max_rad]);
            measured(k) = true;
            fprintf('init_arm: J%d limits from file: [%.1f %.1f] deg\n', ...
                    k, rangeDeg(k,1), rangeDeg(k,2));
        end
    end
else
    fprintf(['init_arm: no %s — using placeholder +-90 deg limits. ' ...
             'Measure with scripts/find_joint_limits.py.\n'], limitsFile);
end
if any(~measured)
    missing = sprintf(' J%d', find(~measured));
    fprintf(['init_arm: *** NO MEASURED LIMITS for%s -- these carry the wide ' ...
             '+-90 deg placeholder,\n' ...
             'init_arm:     so IK will preferentially spend them over joints ' ...
             'that ARE limited.\n' ...
             'init_arm:     Measure with: python scripts/find_joint_limits.py ' ...
             '--joint N\n'], missing);
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

%% ===== PART 3: GEOMETRY SELF-CHECK =====
%
% WHY THIS EXISTS. revNums above maps servo k onto "Revolute <n>" in the
% Simscape model, and it is a HARDCODED GUESS. The assert after it only checks
% that six blocks were found -- never that they are the right six, or that they
% come in chain order. A wrong mapping produces a fully working solver that
% describes a different arm, which is indistinguishable from a correct one until
% somebody measures the physical machine.
%
% That is not hypothetical. On 2026-08-07 a jog measured the claw sweeping
% 30.0 mm where the model predicted 13.8 (x2.17): the model puts the tip 63 mm
% from J2's axis when the arm really has it at ~137 mm. Three sessions of
% calibration work went into servo_calibration.json before anyone checked the
% model's own geometry, and every one of those parameters turned out to be
% correct.
%
% The invariant below needs no ruler and no hardware. For a serial arm the tip's
% perpendicular distance to a joint's axis is what turns that joint's rotation
% into tip travel (sweep = 2*r*sin(theta/2)). J2 carries J3, J4, J5 and the claw,
% so in an ordinary posture its arm must be the LARGEST of the pitch joints. A
% downstream joint reporting a bigger arm is not proof on its own -- a folded arm
% can swing the tip back toward an upper axis -- but at the ZERO configuration
% checked here it means the chain order is wrong.
cfg0chk  = homeConfiguration(robot);
T_tipchk = getTransform(robot, cfg0chk, endEffector);
p_tip    = T_tipchk(1:3,4);

fprintf('init_arm: servo -> model mapping (was silent until 2026-08-07):\n');
armR = zeros(1,5);
for k = 1:5
    Tb   = getTransform(robot, cfg0chk, motorBodies{k});
    ax_l = robot.Bodies{motorIdx(k)}.Joint.JointAxis(:);
    n    = Tb(1:3,1:3) * (ax_l / norm(ax_l));
    v    = p_tip - Tb(1:3,4);
    armR(k) = norm(v - (v.'*n)*n);
    fprintf('init_arm:   servo %d -> Revolute %-2d  body %-10s  tip %6.1f mm from its axis\n', ...
            k, revNums(k), motorBodies{k}, 1000*armR(k));
end

bad = find(armR(3:4) > armR(2)) + 2;
if ~isempty(bad)
    fprintf(2, ['init_arm: *** GEOMETRY WARNING. J%s has a LARGER moment arm to the\n' ...
                'init_arm:     claw than J2, which is upstream of it. At the zero\n' ...
                'init_arm:     configuration that means revNums maps the servos onto\n' ...
                'init_arm:     the wrong Revolute blocks, or the imported link\n' ...
                'init_arm:     transforms are wrong. FK will be confidently incorrect.\n' ...
                'init_arm:     Check with: python scripts/audit_model_axes.py (section D)\n'], ...
            strjoin(arrayfun(@(j) sprintf('%d', j), bad, 'UniformOutput', false), ', '));
end

% Heights a ruler can check at the home pose, printed so a mismatch is caught on
% the day rather than inferred from a failed pick weeks later. MODEL frame here:
% model +Z is physically DOWN (see CLAUDE.md), so these are heights BELOW the
% base origin on the real arm, and the claw tip must come out on the far side of
% the wrist from the shoulder.
T_sh = getTransform(robot, cfg0chk, motorBodies{2});
T_el = getTransform(robot, cfg0chk, motorBodies{3});
T_wr = getTransform(robot, cfg0chk, motorBodies{5});
fprintf(['init_arm: model-frame z at home -- shoulder %.1f, elbow %.1f, ' ...
         'wrist %.1f, claw tip %.1f mm\n'], ...
        1000*T_sh(3,4), 1000*T_el(3,4), 1000*T_wr(3,4), 1000*p_tip(3));
