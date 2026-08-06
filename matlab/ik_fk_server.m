%% =========================================================================
%  IK/FK SERVER - TCP JSON interface wrapping the validated IKtrials_v2 logic
%
%  Listens on localhost:9999 for newline-delimited JSON requests:
%    IK:  {"cmd":"ik","x":...,"y":...,"z":...}
%         -> {"ok":true,"angles_rad":[j1..j5],"err_mm":...} or {"ok":false,"error":"..."}
%    FK:  {"cmd":"fk","angles_rad":[a1,a2,a3,a4,a5]}
%         -> {"ok":true,"T":[16 floats, row-major]}   (T is the WRIST, Body08)
%
%  Implemented on raw java.net.ServerSocket/Socket, NOT MATLAB's `tcpserver`
%  object -- `tcpserver` requires Instrument Control Toolbox, which is not
%  licensed on every machine this needs to run on. MATLAB's own JVM exposes
%  java.net.* with no extra toolbox, and the wire protocol (newline-delimited
%  JSON) is unchanged, so matlab_client.py on the Python side needs no changes.
%
%  RUN THIS FROM THE matlab/ FOLDER (init_arm cd's here anyway). It blocks in
%  an accept() loop, serving one client connection at a time (matches how
%  MatlabIKClient uses it: one persistent connection per HardwareRobot
%  instance) -- when a client disconnects it goes back to waiting for the
%  next one. Hit Ctrl+C in the MATLAB command window to stop it.
%
%  The arm model + IK solver are built ONCE by init_arm.m (shared with the
%  offline harness test_ik_fk.m); every request is then solved fresh against
%  that fixed model -- no per-call state, so identical inputs give identical
%  outputs regardless of call history.
% =========================================================================

clear; clc;

% Build robot / ik / motorIdx / homeAngles / endEffector / wristBody /
% maxReach / IK_TOL as globals. init_arm is a script so its placeholder
% signals + smiData land in this (base) workspace for importrobot's compile.
fprintf('IK/FK server: initializing...\n');
init_arm;
fprintf('IK/FK server: ready\n');

%% ===== START TCP SERVER (java.net, no toolbox required) =====
PORT = 9999;
serverSocket = java.net.ServerSocket(PORT);
cleanupServer = onCleanup(@() serverSocket.close());  %#ok<NASGU>
fprintf('Listening on localhost:%d (Ctrl+C to stop)...\n', PORT);

while true
    clientSocket = [];
    try
        fprintf('Waiting for a connection...\n');
        clientSocket = serverSocket.accept();
        fprintf('Client connected: %s\n', char(clientSocket.getInetAddress().toString()));

        in = java.io.BufferedReader(java.io.InputStreamReader(clientSocket.getInputStream()));
        out = java.io.PrintWriter(clientSocket.getOutputStream(), true);  % autoFlush=true

        while true
            lineJava = in.readLine();
            % EOF is a Java null, which arrives as []. A BLANK line arrives as a
            % zero-length java.lang.String, which is also isempty() -- so test for
            % null specifically, or a stray blank line would drop the connection
            % instead of being skipped by the strlength guard below.
            if isempty(lineJava) && ~ischar(lineJava) && ~isa(lineJava, 'java.lang.String')
                break;  % client closed the connection
            end
            line = char(lineJava);
            if strlength(line) == 0
                continue;
            end

            try
                req = jsondecode(line);
                if strcmp(req.cmd, 'ik')
                    % Optional seed_rad = the arm's CURRENT joint angles, so the
                    % solver returns the nearest solution rather than any legal
                    % one. Absent -> fall back to homeConfiguration.
                    %
                    % Optional lock = joint numbers (1..5) to HOLD at their seed
                    % angle. A 5-DOF arm solving a 3-DOF position target has a
                    % 2-dimensional null space, and nothing in a position-only
                    % solve prefers one point in it over another. Locking spends
                    % that redundancy deliberately instead of leaving it to the
                    % solver -- see handle_ik_request.
                    if isfield(req, 'seed_rad'), sd = req.seed_rad; else, sd = []; end
                    if isfield(req, 'lock'), lk = req.lock; else, lk = []; end
                    resp = handle_ik_request(req.x, req.y, req.z, sd, lk);
                elseif strcmp(req.cmd, 'fk')
                    resp = handle_fk_request(req.angles_rad);
                else
                    resp = struct('ok', false, 'error', 'unknown command');
                end
            catch ME
                % Never let a bad/unreachable request kill the connection
                resp = struct('ok', false, 'error', ME.message);
            end

            out.println(jsonencode(resp));
        end

        fprintf('Client disconnected.\n');
        clientSocket.close();

    catch ME
        fprintf('Connection error: %s\n', ME.message);
        if ~isempty(clientSocket)
            try
                clientSocket.close();
            catch
            end
        end
    end
end

function resp = handle_ik_request(x, y, z, seed_rad, lockJoints)
    global robot ik motorIdx homeAngles endEffector maxReach IK_TOL

    p_robot = [x; y; z];

    % Reach clamp
    d = norm(p_robot);
    if d > maxReach
        resp = struct('ok', false, 'error', sprintf(...
            'target %.0f mm > reach %.0f mm', 1000*d, 1000*maxReach));
        return;
    end

    targetPose = trvec2tform(p_robot');
    seed = homeConfiguration(robot);
    if ~isempty(seed_rad)
        for k = 1:5
            seed(motorIdx(k)) = seed_rad(k);
        end
    end
    seed0 = seed;

    %% ---- optional joint locking -----------------------------------------
    % THIS ARM IS REDUNDANT FOR THE TASK IT IS ASKED TO DO. Five actuated
    % joints against a 3-DOF position target leaves a 2-dimensional null
    % space, and a position-only solve has no preference within it: every
    % point in that space is an equally correct answer, so the solver returns
    % whichever one its iteration happens to land on. Seeding biases that but
    % does not constrain it.
    %
    % That redundancy is not free, because the camera rides on the wrist.
    % Base yaw in particular pans the entire image, so a solve that quietly
    % spends a couple of degrees of J1 to reach the same point disturbs the
    % visual loop more than the move it was asked for. Locking lets the
    % CALLER spend the redundancy deliberately: a pure descent can say "hold
    % J1" and get an answer in the plane the arm is already in.
    %
    % Implemented by pinning the joint's PositionLimits around its seed
    % value, which is the only mechanism rigidBodyTree offers. Restored on
    % every exit path via onCleanup -- `robot` is a handle object and a
    % leaked pin would silently narrow the workspace for every later request.
    restore = {};
    if ~isempty(lockJoints)
        if isempty(seed_rad)
            resp = struct('ok', false, 'error', ...
                'lock requires seed_rad: a joint can only be held at a known angle');
            return;
        end
        LOCK_EPS = 1e-6;
        for k = reshape(double(lockJoints), 1, [])
            if k < 1 || k > 5, continue; end
            i = motorIdx(k);
            jnt = robot.Bodies{i}.Joint;
            % HomePosition is saved TOO, and that is not bookkeeping padding.
            % See below: pinning mutates it, and restoring PositionLimits alone
            % leaves the mutation in place for the rest of the server's life.
            restore{end+1} = {i, jnt.PositionLimits, jnt.HomePosition}; %#ok<AGROW>
            v = seed(i);
            % Clamp the pin INTO the joint's real limits. A seed outside them
            % (a joint that has sagged past its stop, which happens on this
            % arm) would otherwise produce an empty interval and an
            % unsolvable problem reported as "outside workspace".
            v = min(max(v, jnt.PositionLimits(1)), jnt.PositionLimits(2));

            % HOME FIRST, THEN LIMITS. Setting PositionLimits to a band that
            % excludes the joint's current HomePosition makes rigidBodyJoint
            % warn and silently reset HomePosition to the band centre -- and
            % that reset is NOT undone by putting PositionLimits back. Every
            % locked solve was leaving a motor joint's HomePosition wherever
            % the arm happened to be, so homeConfiguration(robot) stopped
            % returning the arm's real zero. Setting it deliberately first
            % makes the change ours, and it is restored on the way out.
            jnt.HomePosition = v;

            % A NARROW BAND, NOT A POINT. [v, v] is a degenerate interval and
            % the solver does not reliably respect it -- observed 2026-08-06,
            % a run that locked J5 still came back wanting 94 and then -100
            % ticks of it. init_arm.m freezes the idler disks with
            % [h - eps, h + eps] for exactly this reason; matching that is the
            % pattern already proven on this model.
            jnt.PositionLimits = [v - LOCK_EPS, v + LOCK_EPS];
        end
    end
    cleanupObj = onCleanup(@() restore_joint_limits(restore)); %#ok<NASGU>

    %% ---- multi-restart, preferring the NEAREST posture -------------------
    % Seeded from the arm's current joint angles: a solver seeded elsewhere
    % returns a valid solution in a completely different posture -- for a
    % 45mm move that once showed up as ~2600 ticks (~227deg) of commanded
    % base rotation.
    %
    % The restart fallback used to throw that away. When the seeded attempt
    % missed by >2mm it reseeded with randomConfiguration and then accepted
    % whichever candidate had the LOWEST POSITION ERROR, regardless of
    % posture -- so a 0.1mm solution half a workspace away beat a 3mm one
    % right where the arm stood, well inside the 10mm tolerance that governs
    % acceptance anyway. Measured 2026-08-05: a 40mm descent came back
    % wanting 213 ticks of J1 and 1275 ticks of J5, while 5/10/20mm descents
    % from the same pose needed 1.2 ticks of J1.
    %
    % So: among candidates that MEET the tolerance, take the one closest in
    % joint space to where the arm actually is. Accuracy beyond IK_TOL buys
    % nothing this arm can execute -- its open-loop positioning error is
    % several millimetres -- whereas posture change is paid for in real
    % motion, real time, and a swung camera.
    weights = [0 0 0 1 1 1];  % position only
    sols = {}; errs = []; moves = [];
    for attempt = 1:10
        [sol, ~] = ik(endEffector, targetPose, weights, seed);
        T = getTransform(robot, sol, endEffector);
        err = norm(T(1:3,4) - p_robot);
        move = max(abs(sol(motorIdx(1:5)) - seed0(motorIdx(1:5))));

        sols{end+1} = sol; errs(end+1) = err; moves(end+1) = move; %#ok<AGROW>

        % The seeded attempt landing inside tolerance is the answer we want:
        % it is both accurate enough and, by construction, the nearest
        % posture. Restarting from there could only find something further
        % away that scores no better.
        if attempt == 1 && err <= IK_TOL
            break;
        end
        seed = randomConfiguration(robot);
    end

    good = find(errs <= IK_TOL);
    if isempty(good)
        [bestErr, j] = min(errs);
        resp = struct('ok', false, 'error', sprintf(...
            'IK missed by %.1f mm (> %.0f mm tol). Target outside workspace.', ...
            1000*bestErr, 1000*IK_TOL));
        return;
    end
    [~, jj] = min(moves(good));
    j = good(jj);
    bestSol = sols{j}; bestErr = errs(j); bestMove = moves(j);

    % Extract J1..J5 angles (J6 always home, manually commanded)
    ik_angles = homeAngles;
    for k = 1:5
        ik_angles(k) = bestSol(motorIdx(k));
    end

    % move_rad lets the caller see how much posture this solve costs, which
    % a position residual cannot show: the 213-tick J1 solve reported 0.0 mm.
    %
    % lock_drift_rad is how far the LOCKED joints actually moved, and it exists
    % because a lock that silently fails is worse than no lock: the caller
    % believes a disturbance is suppressed and tunes against that belief. On
    % 2026-08-06 a run locking J5 came back moving it 94 ticks and nothing in
    % the protocol could say so -- the pin was a degenerate [v, v] interval the
    % solver did not honour. Callers should treat a non-trivial value here as a
    % failed lock, not as noise.
    lockDrift = 0;
    for k = reshape(double(lockJoints), 1, [])
        if k < 1 || k > 5, continue; end
        lockDrift = max(lockDrift, abs(bestSol(motorIdx(k)) - seed0(motorIdx(k))));
    end

    resp = struct('ok', true, 'angles_rad', ik_angles(1:5), ...
                  'err_mm', 1000*bestErr, 'move_rad', bestMove, ...
                  'lock_drift_rad', lockDrift);
end

function restore_joint_limits(restore)
    % Put back every PositionLimits AND HomePosition that locking touched. Runs
    % on normal return AND on error, because `robot` is a global handle object:
    % a leaked pin would silently freeze that joint for every later request in
    % this server's lifetime, which would look like an arm that had lost a
    % degree of freedom for no reason.
    %
    % HomePosition is restored SECOND, and the order matters: widening
    % PositionLimits first means the home value is legal by the time it is
    % written, so putting it back cannot itself trip the reset-and-warn path
    % that made it necessary.
    global robot
    for n = 1:numel(restore)
        jnt = robot.Bodies{restore{n}{1}}.Joint;
        jnt.PositionLimits = restore{n}{2};
        jnt.HomePosition = restore{n}{3};
    end
end

function resp = handle_fk_request(angles_rad)
    global robot motorIdx homeAngles wristBody endEffector

    % Build config: start from home, override J1..J5
    cfg = homeConfiguration(robot);
    for k = 1:5
        cfg(motorIdx(k)) = angles_rad(k);
    end

    % Two DIFFERENT frames, both needed, ~CLAW_LEN (70mm) apart:
    %   T     = wrist (Body08) -- what hand-eye calibration is solved against,
    %           so this is the one the vision pipeline composes with
    %           T_gripper_camera. Must stay first/unchanged for compatibility.
    %   T_tip = ClawTip -- what the IK solver actually targets. Round-trip
    %           validation (command a target, move, read back, compare) has to
    %           compare against THIS, or it measures a 70mm frame offset
    %           instead of real positioning error. Estimating it as
    %           "wrist minus 70mm down" only holds while the claw points
    %           straight down, which stops being true as the arm tilts.
    T = getTransform(robot, cfg, wristBody);
    T_tip = getTransform(robot, cfg, endEffector);

    % Return as row-major 16-element arrays
    resp = struct('ok', true, ...
                  'T', reshape(T', 1, 16), ...
                  'T_tip', reshape(T_tip', 1, 16));
end
