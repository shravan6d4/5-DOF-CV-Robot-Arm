function [robot, motorIdx, endEffector, wristBody] = build_arm_from_survey()
%BUILD_ARM_FROM_SURVEY  A rigidBodyTree built from a RULER SURVEY, not from CAD.
%
%   [robot, motorIdx, endEffector, wristBody] = build_arm_from_survey()
%
%   NOT WIRED IN. init_arm.m still imports Robomainassemjoints.slx and nothing
%   calls this yet. It exists so the corrected geometry has a MATLAB form ready
%   to swap in, and so test_arm_from_survey.m can check it against measured data.
%
%   WHY IT EXISTS. importrobot produces an arm that is the wrong shape. Measured
%   2026-08-07 at the home pose, heights above the tabletop:
%
%       shoulder     model 158.2   ruler  90     off by -68 mm
%       elbow        model 106.4   ruler 146     off by +40
%       wrist pitch  model 108.0   ruler 152     off by +44
%       wrist        model  71.3   ruler  71.3   exact
%       claw tip     model   5.5   ruler   0
%
%   The shoulder settles it: that shaft is bolted to the base column, so no joint
%   angle, dir_sign, ticks_per_rad or backlash can move it, and the model reports
%   158.2 at every pose. The imported model puts the elbow 51.8 mm BELOW the
%   shoulder where the arm has it 56 mm ABOVE -- the upper arm points ~30 deg
%   down in CAD and ~33 deg up in reality. That is why a J2 jog swept the claw
%   30.0 mm where the model predicted 13.8.
%
%   WHY THE SURVEY IS ENOUGH. J2, J3 and J4 are parallel (audit_model_axes.py),
%   so the arm is base-yaw + a planar 3-link chain + a wrist roll. Five points in
%   that plane determine it outright, with no fitting of any kind. This mirrors
%   src/vision_pipeline/kinematics/arm_model.py exactly; the two must agree, and
%   test_arm_from_survey.m checks this one against the same held-out touches
%   tests/test_arm_model.py uses (6.0 mm spread, against the CAD model's 73.1).
%
%   FRAME. PHYSICAL, and deliberately NOT the imported model's frame:
%       +X  the arm's forward at yaw zero      +Y  left      +Z  up
%   The origin sits ON THE BASE YAW AXIS at TABLE level, because that is what the
%   survey was measured from and what physically exists. The imported model's
%   origin is a CAD artefact 81 mm off that axis, and its +Z points physically
%   DOWN. So a server using this tree must NOT apply the diag(1,-1,-1,1) flip
%   MatlabIKClient does -- this tree is already physical. See CLAUDE.md
%   "COORDINATE FRAMES".
%
%   LIMITS, stated because this will be trusted:
%     * The survey is 2-D. Any LATERAL offset of the claw from the arm's plane is
%       unmeasured and taken as zero. It biases Y, never height, so it cannot
%       affect the touch validation.
%     * Height is what the touches validate. The forward coordinate carries only
%       the survey's own accuracy and nothing independent has checked it.
%     * Gear backlash is not modelled and is the likeliest source of the
%       residual 6 mm.

%% ---- the survey -------------------------------------------------------------
% Operator ruler measurements at HOME (all five joint angles zero), 2026-08-07.
% [forward from servo 1's shaft, up from the tabletop], millimetres.
%
% Note the elbow sits BEHIND the base column at home (-72) while the wrist pitch
% is well in front (+67): at home this arm folds back over itself, upper arm
% up-and-back and forearm forward. That is the posture CAD gets inside-out.
S.shoulder    = [ 12.0,  90.0];    % servo 2 shaft
S.elbow       = [-72.0, 146.0];    % servo 3 shaft
S.wrist_pitch = [ 67.0, 152.0];    % servo 4 shaft
S.wrist       = [ 74.0,  71.3];    % servo 5 shaft -- the hand-eye frame
S.tip         = [ 75.0,   0.0];    % claw tip, ON the table at home

names = {'shoulder','elbow','wrist_pitch','wrist','tip'};

% Segment vectors in the arm's plane. At the zero configuration every body frame
% is axis-aligned with the base, so each link's fixed transform is a PURE
% TRANSLATION by its segment -- no rotations to get wrong.
seg = zeros(4,2);
for k = 1:4
    seg(k,:) = S.(names{k+1}) - S.(names{k});
end

%% ---- axes ------------------------------------------------------------------
% Yaw about vertical.
YAW_AXIS = [0 0 1];

% Pitch. In a right-handed frame with +X forward and +Z up, a rotation carrying
% +X toward +Z (which is how arm_model.py measures its angles, atan2(dz, dx)) is
% a rotation about -Y. Getting this backwards inverts every pitch joint, so it is
% named rather than inlined and test_arm_from_survey.m is what proves it.
PITCH_AXIS = [0 -1 0];

% Wrist ROLL, about the claw's own direction. NOT the forearm: audit_model_axes
% reported J5 as 0.2 deg off the forearm, but that reading came from the CAD this
% file replaces. The operator's jog is the evidence that counts -- "the claw
% SPINS ANTICLOCKWISE viewed from the wrist looking out along the claw" -- so the
% axis runs wrist->tip. A consequence worth noting: the tip then lies ON that
% axis, so J5 cannot move it, and only a lateral offset would make it do so.
roll_dir = [seg(4,1), 0, seg(4,2)];
ROLL_AXIS = roll_dir / norm(roll_dir);

%% ---- build -----------------------------------------------------------------
robot = rigidBodyTree('DataFormat', 'row', 'MaxNumBodies', 8);
robot.BaseName = 'base';

% Joint k rotates the frame at survey point k. J1 sits at the origin (the yaw
% axis passes through it by construction); the rest step along the chain.
jointAxes  = {YAW_AXIS, PITCH_AXIS, PITCH_AXIS, PITCH_AXIS, ROLL_AXIS};
jointNames = {'J1','J2','J3','J4','J5'};
bodyNames  = {'link1','link2','link3','link4','link5'};

% Translations INTO each joint frame, expressed in the parent's frame.
offsets = [ 0, 0, 0;                                   % base   -> J1 (yaw axis)
            S.shoulder(1), 0, S.shoulder(2);           % J1     -> J2 (shoulder)
            seg(1,1), 0, seg(1,2);                     % J2     -> J3 (elbow)
            seg(2,1), 0, seg(2,2);                     % J3     -> J4 (wrist pitch)
            seg(3,1), 0, seg(3,2) ] / 1000;            % J4     -> J5 (wrist)

parent = 'base';
motorIdx = zeros(1,5);
for k = 1:5
    b = rigidBody(bodyNames{k});
    j = rigidBodyJoint(jointNames{k}, 'revolute');
    j.JointAxis = jointAxes{k};
    setFixedTransform(j, trvec2tform(offsets(k,:)));
    % Home is angle ZERO on every joint, by construction: the survey was taken at
    % the pose servo_calibration.json calls home, so ticks_to_rad returns 0 there
    % and the tree must agree without an offset.
    j.HomePosition = 0;
    b.Joint = j;
    addBody(robot, b, parent);
    parent = bodyNames{k};
    motorIdx(k) = robot.NumBodies;
end

% Claw tip: fixed, hanging from the wrist. Same segment the roll axis came from.
tipBody = rigidBody('ClawTip');
tipJoint = rigidBodyJoint('ClawTipJnt', 'fixed');
setFixedTransform(tipJoint, trvec2tform([seg(4,1), 0, seg(4,2)] / 1000));
tipBody.Joint = tipJoint;
addBody(robot, tipBody, 'link5');

endEffector = 'ClawTip';
wristBody   = 'link5';      % servo 5's shaft: the frame hand-eye solves against
end
