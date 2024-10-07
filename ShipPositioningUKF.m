%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% AUTHOR Benjamin Noack
% Based on Florian Schiegg's Pyhon Code
%
% Date 24.05.2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
format long g % better readability for debugging
% rng(11) % remove to make it random again

sensors_latlon = horzcat([59.5376, 10.5568]', [59.5789, 10.6104]', [59.5881, 10.6530]', [59.5327, 10.6619]');

sensors_latlon = horzcat(...
    [59.54421451622937, 10.56664697057754]', ...
    [59.58296252718586, 10.61733019341999]', ...
    [59.57576969007426, 10.64846570881250]', ...
    [59.43593598670043, 10.58136009140111]'...
    );

z1 = utmzone(sensors_latlon');
[ellipsoid,estr] = utmgeoid(z1);
utmstruct = defaultm('utm');
utmstruct.zone = z1;
utmstruct.geoid = ellipsoid;
utmstruct = defaultm(utmstruct);


[x,y] = projfwd(utmstruct,sensors_latlon(1,:),sensors_latlon(2,:));
% sensors = struct('x',num2cell(x),'y',num2cell(y));
% sensors = struct('pos',num2cell([x;y],1));
sensors.pos = [x;y];

% simulation times
% n_time_steps = 50000;
dt = 0.001*0.05; % 0.05 ms
n_time_steps = 300/dt; % simulate 300 sec
time_steps = dt*(1:n_time_steps);

% sensors' parameters

% sensors.vel = -2 * pi * 0.5*ones(size(sensors)); % 30 rpm 
% sensors.length = 8000;
 %zeros(1,n_time_steps);
sensors.last_delta_angles = zeros(1,length(sensors.pos));%repmat(mod(sensors.t*sensors.vel,2*pi),4,1); 
sensors.peaks = nan(length(sensors.pos),1);
sensors.time_step_of_measurement = nan(length(sensors.pos),1); % discrete time step
sensors.z = nan(length(sensors),n_time_steps);
% number_of_peaks_used_for_LS is number of measurements used for each LS
% estimate
% minimum value: 4
% all measurements: length(sensors.pos)
% all plus difference between first and last meas.: length(sensors.pos)+1
sensors.number_of_peaks_used_for_LS = length(sensors.pos);% length(sensors.pos)


ship.pos = mean(sensors.pos,2);
% ship.vel = [-0.16, -0.08]'/dt;
vel = 24 / 1.944; %convvel(24,'kts','m/s'); % 24 Knoten in m/s
heading = deg2rad(-140); 
ship.vel = [cos(heading);sin(heading)]*vel;
ship.trajectory = zeros(2,n_time_steps);
ship.radar.vel = -2 * pi * 0.5; % 30 rpm
ship.radar.length = 8000;
ship.radar.angles = mod((time_steps-dt)*ship.radar.vel,2*pi);

% add const. turn rate to true ship model
ship.turn_rate = 0*dt; % degree per s
ship.turn_R = [cosd(ship.turn_rate) -sind(ship.turn_rate);sind(ship.turn_rate) cosd(ship.turn_rate)];


% % Sensor placement does not seem to have a meaning in real world
% wgs84 = wgs84Ellipsoid('kilometer');
% [x,y,z] = geodetic2ecef(wgs84,sensors_latlon(1,:),sensors_latlon(2,:),0);
% axesm utm
% setm(gca,'zone',z1)
% h = getm(gca);
% setm(gca,'grid','on','meridianlabel','on','parallellabel','on')
% load coastlines
% plotm(coastlat,coastlon)
% plotm(sensors_latlon(1,:),sensors_latlon(2,:),'.r','MarkerSize',20)

% Plotting
fig = figure(1);
xlim([min([sensors.pos(1,:)])-500,max([sensors.pos(1,:)])+500])
ylim([min([sensors.pos(2,:)])-500,max([sensors.pos(2,:)])+500])
hold on
h = plot([sensors.pos(1,:)],[sensors.pos(2,:)],'^r','MarkerSize',5);

% Add Legend

qw{1} = plot(nan, '--k');
qw{2} = plot(nan, '-xr','MarkerSize',5);
qw{3} = plot(nan, 'Color', '#0072BD', 'LineWidth', 1, 'LineStyle','-');
legend([qw{:}], {'true pos.','Least Squares','(U)KF'}, 'location', 'northwest','AutoUpdate','off')

% Add Callbacks to update scalebar in the lower right corner
update_scale_bar();
ax=gca();
set(zoom(ax),'ActionPostCallback',@(fig,axes)update_scale_bar());
set(pan(fig),'ActionPostCallback',@(fig,axes)update_scale_bar());



% System Model
Ax = [1,dt;0,1];
A = kron(Ax,eye(2));

% Sensor noise and Process noise
R = 0.1 / 10000; % Variance of R in ms
za = 0.05;
% za = 0.1; %TESTING
Qx = [1/3*dt^2,1/2*dt;1/2*dt,1] * za * dt; %Marchthaler: za in m/s^2 
Q = kron(Qx,eye(2));

h_ship_trajectory = plot(ship.pos(1),ship.pos(2),'--k');

% initialize LS estimate
estLS.x = zeros(2,1);
estLS.P = zeros(2);
estLS.trajectory = nan(2,1);
estLS.k = -1;
h_estLS_trajectory = plot(estLS.x(1),estLS.x(2),'-r','LineWidth',0.25);

% initialize KF estimate
estKF.x = nan(4,1);
estKF.P = zeros(4);
estKF.trajectory = nan(2,1);
estKF.k = -1;

estKF.A = A;
estKF.Q = Q;
estKF.R = 2*R;

% The following is used to initiatize KF estimate
% several LS estimates are collected to get KF prior for pos. and vel. 
estKF.initN = 7;
estKF.initX = [[];[]]; % collects first initN LS estimates to init KF
estKF.initZ = [];
estKF.initH = [];
estKF.initW = [];
estKF.initk = [];

estKF.correlation = zeros(size(estKF.x));


for k = 1:n_time_steps

    % ship Movement
    ship.pos = A(1:2,1:2)*ship.pos + A(1:2,3:4)*ship.vel;
    ship.trajectory(:,k) = ship.pos;
    ship.vel = ship.turn_R*ship.vel;

    % Prediction Step of Filters
    if ~isnan(estKF.x) % wait KF to be initialized
        estKF.x = estKF.A * estKF.x;
        estKF.P = estKF.A * estKF.P * estKF.A' + estKF.Q;
        estKF.k = estKF.k + 1;
        estKF.trajectory(:,estKF.k) = estKF.x;
    end

    ship.beam_end(1) = ship.pos(1) + ship.radar.length * cos(ship.radar.angles(k));
    ship.beam_end(2) = ship.pos(2) + ship.radar.length * sin(ship.radar.angles(k));

    sensor_angles = calc_relative_angle(sensors.pos, repmat(ship.pos,1,length(sensors.pos)));
    delta_angles = sensor_angles-ship.radar.angles(k);

    % Check whether / which sensor detects radar
    if any(sign(delta_angles) ~= sign(sensors.last_delta_angles) & (abs(delta_angles-sensors.last_delta_angles) < 1)) % Second condition to avoid modulo-caused sign changes. Deltas should generally be very close (<<1)
 
        % ID of activated sensor
        id_active_sensor = find(sign(delta_angles) ~= sign(sensors.last_delta_angles) & (abs(delta_angles-sensors.last_delta_angles) < 1));
 
         % mark active sensor in plot
        if exist('h_marked_sensor','var')
            delete(h_marked_sensor)
        end
        h_marked_sensor = plot(sensors.pos(1,id_active_sensor),sensors.pos(2,id_active_sensor),'.','Color','#77AC30','MarkerSize',80);

        fprintf('%0.2f sec: Sensor %d hit! Delta angle: %d\n',k*dt, id_active_sensor, delta_angles(id_active_sensor)*180/pi)

        %measurement
        sensors.peaks(id_active_sensor) = dt*k + mvnrnd(0,R,1); % store current peak time in sec + noise
        sensors.time_step_of_measurement(id_active_sensor) = k; % store time step

        % if all sensors have detected ship once
        if sum(~isnan(sensors.peaks(:))) == length(sensors.pos)

            % remove old circles in plot
            if exist('h_circles','var')
                delete(h_circles)
            end

            % find sensor that has detected ship before
            [~,sorted_ids] = sort(k-sensors.time_step_of_measurement);

            % diff between first and last 
            sorted_ids(end+1) = sorted_ids(1);

            for n = 1:sensors.number_of_peaks_used_for_LS-1
                id_first_sensor = sorted_ids(n);
                first_peak_time = sensors.peaks(id_first_sensor);
                id_second_sensor = sorted_ids(n+1);
                second_peak_time = sensors.peaks(id_second_sensor);

                % ship_angle = (k - previous_peak_time)/2000 * 2*pi;
                ship_angle = (first_peak_time - second_peak_time) * 2*pi / 2; % emitter angle

                ship_angle = mod(ship_angle,pi); % this needs to be checked! Seems to circumvent cases in Python code
                pos_B = sensors.pos(:,id_second_sensor);
                pos_A = sensors.pos(:,id_first_sensor);

                d_sensors = norm(pos_A-pos_B);
                r_circle(n) = d_sensors/sqrt(2-2*cos(2*ship_angle));

                c_circle(n,1) = pos_A(1) + r_circle(n)/d_sensors*((pos_B(1)-pos_A(1))*sin(ship_angle)-(pos_B(2)-pos_A(2))*cos(ship_angle));
                c_circle(n,2) = pos_A(2) + r_circle(n)/d_sensors*((pos_B(1)-pos_A(1))*cos(ship_angle)+(pos_B(2)-pos_A(2))*sin(ship_angle));

                h_circles(n) = viscircles(c_circle(n,:),r_circle(n),'LineStyle',':','Color',[0.5,0.5,0.5],'LineWidth',1);
                uistack(h_circles(n),'bottom',1); % put to background
            end 


            %% LS update

            % measurement mapping, differences of circle centers
            estLS.H = 2*diff(c_circle); 
            % measurements, see eq. (8) in "Scan-Based Emitter Passive
            % Localization
            estLS.z = diff(vecnorm(c_circle').^2)' - diff(r_circle.^2)';

            % n_12-n_23 correspond to measurement noise 2R, correlations
            % are then R
            n = length(c_circle)-1;

            W = diag(2*R*ones(1,n)) + diag(-R*ones(1,n-1),1) + diag(-R*ones(1,n-1),-1);      
            % THIS IS NOT THE CORRECT W FOR THESE MEASUREMENT EQUATIONS AS
            % IT ENTERS THE VAR ship_angle
            % estLS.x = (estLS.H'/W*estLS.H)\estLS.H'/W*estLS.z;
            
            estLS.x = (estLS.H'*estLS.H)\estLS.H'*estLS.z;
            if isnan(estLS.trajectory)
                estLS.trajectory = [estLS.x,zeros(length(estLS.x),n_time_steps-k-1)];
                estLS.k = 1;
            else
                estLS.k = estLS.k + 1;
                estLS.trajectory(:,estLS.k) = estLS.x;
            end
            plot(estLS.x(1),estLS.x(2),'xr','MarkerSize',5)


            %% Initialize KF with first initN LS estimate(s)
            if isnan(estKF.x) % wait to initialize KF
                if estKF.initN == 1 % in this case no vel. can be estimated
                    estKF.x = [estLS.x;0;0];
                    % estKF.x = [estLS.x;ship.vel];
                    estKF.P = 10*blkdiag(inv(estLS.H'/W*estLS.H),100*eye(2));
                    h_estKF_trajectory = plot(estKF.x(1),estKF.x(2),'-b');
                    estKF.trajectory = [estKF.x,zeros(length(estKF.x),n_time_steps-k-1)];
                    estKF.k = 1;
                elseif estKF.initN == length(estKF.initk)+1
                    % NOTE: THIS IS IMPLEMENTED VERY INEFFICIENT AND SLOPPY
                    % IGNORING SOME CORRELATIONS

                    estKF.initX = [estLS.x estKF.initX ];
                    dk = [0, k - estKF.initk];

                    % Use an FIR estimate of horizon initN to initalize KF
                    HFIR = [];
                    Hpos = [eye(length(estLS.x)),zeros(length(estKF.x)-length(estLS.x))];
                    
                    for i = 1:estKF.initN
                        HFIR = [HFIR;Hpos/(estKF.A^dk(i))];               
                    end


                   %  % improve implementation and does WFIR improve anything?
                   %  WFIR = repmat({zeros(2)},estKF.initN,estKF.initN); % see, e.g., FUSION22_Oehl, Vk
                   %  for i = 1:estKF.initN          
                   %      for j = 1:i
                   %          WFIR{i,j} = WFIR{i,j} + Hpos*inv(estKF.A^dk(i))*estKF.Q * inv(estKF.A^dk(j)) * Hpos';
                   %          if i ==j
                   %              WFIR{i,j} = WFIR{i,j} +  inv(estLS.H'/W*estLS.H); % add "meas. noise"
                   %          else
                   %              WFIR{j,i} = WFIR{i,j}';
                   %          end
                   %      end      
                   %  end
                   % 
                   % WFIR = cell2mat(WFIR);
                    

                    zFIR = reshape(estKF.initX,[],1);

                    % INIT WITHOUT WEIGHTING
                    estKF.x = (HFIR'*HFIR)\HFIR'*zFIR;
                    estKF.P = 2000*inv(HFIR'*HFIR);

                    estKF.x(3:4) = zeros(1,2);

                    % %% INIT WITH WEIGHTING
                    % estKF.x = (HFIR'/WFIR*HFIR)\HFIR'/WFIR*zFIR;
                    % % estKF.P = 100*blkdiag(inv(estLS.H'*estLS.H),eye(2));
                    % estKF.P = blkdiag(200*eye(2),50*eye(2))+inv(HFIR'/WFIR*HFIR); % increase uncertainty

           
                    disp('Initializing Kalman Filter')
                    
                    h_estKF_trajectory = plot(estKF.x(1),estKF.x(2),'-b');
                    estKF.trajectory = [estKF.x,zeros(length(estKF.x),n_time_steps-k-1)];
                    estKF.k = 1;
                else   
                    estKF.initX = [estLS.x estKF.initX ];
                    % estKF.initZ = [estLS.z; estKF.initZ];
                    % % estKF.initH = [estLS.H; estKF.initH];
                    % estKF.initH = blkdiag([estLS.H, zeros(length(estLS.H),(length(estKF.x)-length(estLS.x)))],estKF.initH);
                    % estKF.initW = blkdiag(W, estKF.initW);
                    estKF.initk = [k, estKF.initk];
                end

            else%if mod(estKF.k,2)==0
                %% Update KF using UKF filtering step
                id1 = sorted_ids(1); id2 = sorted_ids(2);
                x1 = sensors.pos(:,id1); x2 = sensors.pos(:,id2);

                % FORMULATION ACCORDING TO (13)
                z12 = -(sensors.peaks(id1) - sensors.peaks(id2)); % z12 = t12 in (13), CHECK SIGN

                r1 = @(x)vecnorm(x-x1); r2 = @(x)vecnorm(x-x2);
                h = @(x) acos((r1(x).^2+r2(x).^2-norm(x1-x2)^2)./(2*r1(x).*r2(x)))/ship.radar.vel; % speed of signal in (13) is ignored (not modeled)
    

                % Compute correlations with previous estimate / NEEDS TO BE
                % CHECKED AGAIN
                % corr = A^elapsed_time_steps*K_k*R
                meascorr = estKF.A^(sensors.time_step_of_measurement(id1)-sensors.time_step_of_measurement(id2))*estKF.correlation;
                joint_cov = blkdiag(estKF.P,2*estKF.R);
                joint_cov(1:end-1,end) = meascorr;
                joint_cov(end,1:end-1) = meascorr';

                [s,w] = sigma_points([estKF.x;0],joint_cov);

                sz = h(s(1:2,:)) + s(length(estKF.x)+1:end,:); % sigma points for measurements, h only applies to the pos. components of state

                zpred = sz*w';

                sqxx = sqrt(w).*(s(1:length(estKF.x),:)-estKF.x);
                
                sqzz = sqrt(w).*(sz-zpred);
                Czz = sqzz*sqzz';
                Cxz = sqxx*sqzz';

                Kk = Cxz/Czz;
                estKF.x = estKF.x + Kk*(z12 - zpred);
                estKF.P = estKF.P - Kk*Czz*Kk'; 

                estKF.trajectory(:,estKF.k) = estKF.x;

                % store correlation for next update
                estKF.correlation = -Kk*estKF.R;

                fprintf('[KF]  max. standard deviation in position: %0.2f m\n',sqrt(max(eig(estKF.P(1:2,1:2)))))

            end

        end

    end

    sensors.last_delta_angles = delta_angles;
   

   % update plot
   if mod(k*dt,0.01) == 0
       if exist('h_radar_beam','var')
           delete(h_radar_beam)
           delete(h_ship)
       end

       if exist('h_marked_sensor','var')
          h_marked_sensor.MarkerSize =  h_marked_sensor.MarkerSize*0.90;
       end

       % current position and radar beam
       h_radar_beam = plot([ship.pos(1), ship.beam_end(1)], [ship.pos(2), ship.beam_end(2)], 'Color','#77AC30');
       h_ship = plot(ship.pos(1), ship.pos(2), 'ok', 'MarkerSize',5);

       % update trajectory
       h_ship_trajectory.XData = ship.trajectory(1,1:k);
       h_ship_trajectory.YData = ship.trajectory(2,1:k);

       % update trajectory generated from LS estimate
       h_estLS_trajectory.XData = estLS.trajectory(1,1:estLS.k);
       h_estLS_trajectory.YData = estLS.trajectory(2,1:estLS.k);

       if exist('h_KF','var')
           delete(h_KF)
           delete(h_KF_ell)
       end
       if ~isnan(estKF.x)
          h_KF = plot (estKF.x(1),estKF.x(2),'.','Color','#0072BD','MarkerSize',20);
          ell = ellipse( estKF.x(1:2), estKF.P(1:2,1:2), 1, 50 );
          h_KF_ell = line(ell(1,:), ell(2,:),'Color', '#0072BD', 'LineWidth', 1, 'LineStyle','-');
          
          h_estKF_trajectory.XData = estKF.trajectory(1,1:estKF.k);
          h_estKF_trajectory.YData = estKF.trajectory(2,1:estKF.k);
       end

       drawnow
       pause(0.01)
   end

end
hold off


%%% Functions

% relative angle between sensor and ship
function angle = calc_relative_angle(sensor, ship)
    angle = mod(atan2(sensor(2,:)-ship(2,:), sensor(1,:)-ship(1,:)),2*pi); %(np.pi*2)
end

% Unscented Transform for UKF
function [s,w] = sigma_points(x,C, w0) % -1 < w0 < 1
    dimx = length(x);
    if nargin < 3
        w0 = 1/3;
    end
    
    S = sqrt(dimx/(1-w0))*sqrtm(C); %chol(C,'lower');
    T = [zeros(dimx,1),eye(dimx),-eye(dimx)];
    s = bsxfun(@plus, S*T, x);

    w = [w0, ones(1,2*dimx)*(1-w0)/(2*dimx)]; % weights for mean

end


% Plotting Helper
% For gif export, speed can be changed with https://ezgif.com/speed (e.g., to 30 %)
function plotsettings(fig)
    % xlim([-1,17])
    % ylim([-3,7])
    set(fig, 'Position', [0 0 1600 1000])
    grid on
    xlabel('x');
    ylabel('y');
end

% update scalebar when axes are zoomed or paned
function update_scale_bar()
   h_sb = findobj('Tag','scalebar');
   delete(h_sb);

   % plot a scalebar
   h_scale = 100;
   hpos = min(xlim) + diff(xlim)*0.9 - [0,h_scale];
   ypos = min(ylim) + diff(ylim)*0.1 * ones(1,2);
   
   toggleback = 0;
   if ~ishold
       hold on
       toggleback = 1;
   end

   plot(hpos,ypos,'k-|','Tag','scalebar');
   text(hpos(1),ypos(1),' 100 m','Tag','scalebar');

   if toggleback
       hold off
   end
end

function X = ellipse( x, C, c, N )
    % COV_ELLIPSE Zeichnet die Ellipse für die Matrix <C> mit
    % Zentrum <x>.
    % IN    x       Mittelwert
    %       C       Matrix der Ellipse
    %       c       Konstante. Größe der Ellipse.                         
    %       N       Anzahl Punkte, die geplottet werden sollen
    % OUT   X       Berechnete Punkte der Eellipse
    % Eigenwertzerlegung
    [B, Bx] = eig(C);
    % L¨nge der Hauptachsen, parametrisch
    a = c*real( sqrt(Bx(1,1)) );
    b = c*real( sqrt(Bx(2,2)) );
    % Punkte auf Ellipse bestimmen
    t = 0:2*pi/N:2*pi;
    z1 = a*cos(t);
    z2 = b*sin(t);
    X = repmat(x, 1, N+1) + B*[z1; z2];
end