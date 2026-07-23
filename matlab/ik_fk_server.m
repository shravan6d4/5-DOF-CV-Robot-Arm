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
                    if isfield(req, 'seed_rad')
                        resp = handle_ik_request(req.x, req.y, req.z, req.seed_rad);
                    else
                        resp = handle_ik_request(req.x, req.y, req.z, []);
                    end
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

function resp = handle_ik_request(x, y, z, seed_rad)
    global robot ik motorIdx homeAngles endEffector maxReach IK_TOL

    p_robot = [x; y; z];

    % Reach clamp
    d = norm(p_robot);
    if d > maxReach
        resp = struct('ok', false, 'error', sprintf(...
            'target %.0f mm > reach %.0f mm', 1000*d, 1000*maxReach));
        return;
    end

    % Multi-restart IK (same as IKtrials_v2 Part 3/4), but seeded from the
    % arm's CURRENT joint angles when the caller supplies them. A solver
    % seeded elsewhere can return a perfectly valid solution in a completely
    % different posture — for a 45mm move that showed up as ~2600 ticks
    % (~227deg) of commanded base rotation. Seeding from where the arm
    % actually is makes the nearest solution the one it converges on.
    targetPose = trvec2tform(p_robot');
    seed = homeConfiguration(robot);
    if ~isempty(seed_rad)
        for k = 1:5
            seed(motorIdx(k)) = seed_rad(k);
        end
    end
    weights = [0 0 0 1 1 1];  % position only
    bestSol = []; bestErr = inf;
    for attempt = 1:10
        [sol, ~] = ik(endEffector, targetPose, weights, seed);
        T = getTransform(robot, sol, endEffector);
        err = norm(T(1:3,4) - p_robot);
        if err < bestErr, bestErr = err; bestSol = sol; end
        if bestErr < 0.002, break; end
        seed = randomConfiguration(robot);
    end

    % Tolerance check
    if bestErr > IK_TOL
        resp = struct('ok', false, 'error', sprintf(...
            'IK missed by %.1f mm (> %.0f mm tol). Target outside workspace.', ...
            1000*bestErr, 1000*IK_TOL));
        return;
    end

    % Extract J1..J5 angles (J6 always home, manually commanded)
    ik_angles = homeAngles;
    for k = 1:5
        ik_angles(k) = bestSol(motorIdx(k));
    end

    resp = struct('ok', true, 'angles_rad', ik_angles(1:5), 'err_mm', 1000*bestErr);
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
