function [output] = dc_opf_solver(data_name, scale)
    % load the case data
    define_constants;
    mpc = loadcase(data_name);
    mpopt = mpoption('verbose', 0, 'out.all', 0);

    % create the uncertainty realization
    W = [];
    shape = size(mpc.bus); % num_bus x num_features
    num_bus = shape(1);

    for row_idx = 1:num_bus
        % uncertainty realization (w)
        % = Normal Dist(mean = 0, std = coeff * Pd( mpc.bus(row_idx, 3) )

        % set the distribution of the uncertainty realization
        mean = 0;
        std = abs(scale * mpc.bus(row_idx, 3));

        % create the uncertainty realization
        w = normrnd(mean, std);

        % append it
        W(end + 1) = w;
    end

    % add the uncertainty to the Pd( mpc.bus(:, 3) )
    mpc.bus(:, PD) = mpc.bus(:, PD) + transpose(W);

    % run dc-opf solver
    results = rundcopf(mpc, mpopt);
    disp(results)

    % store the useful solution info
    p_d = mpc.bus(:, PD); % bus demand active power
    bus_idx = mpc.bus(:, 1); % bus index

    p_g = results.gen(:, PG); % gen active power (PG=2)
    p_g_lim = mpc.gen(:, PMAX:PMIN); % gen active power constraints
    gen2bus = mpc.gen(:, 1); % gen-to-bus mapping

    p_f = results.branch(:, PF); % ??? (PF=14)
    p_f_lim = [mpc.branch(:, RATE_A), -mpc.branch(:, RATE_A)]; % line (branch) constraints: rateA
    bus2bus = mpc.branch(:, 1:2); % fbus-to-tbus mapping

    % set an output structure
    gen_info = struct('p_g', p_g, 'p_g_lim', p_g_lim, 'gen2bus', gen2bus);
    bus_info = struct('p_d', p_d, 'bus_idx', bus_idx);
    flow_info = struct('p_f', p_f, 'p_f_lim', p_f_lim, 'bus2bus', bus2bus);
    w_info = struct('w', W);
    output = struct('bus_info', bus_info, 'gen_info', gen_info, 'flow_info', flow_info, 'w_info', w_info);
    % output = struct('mpc', mpc, 'results', results);

end
