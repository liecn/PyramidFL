clear;
clc;
close all;

%% Set Parameters for Loading Data
lineA = ["-", ":", "--", '-.'];
lineC = ["*", "s", "o", "^", "+", "p", "d"];
lineS = ["-*", "--s", ":^", '-.p'];
color_list=[[0,0,0];[255,128,0];[138, 43, 226];[106,90,205];[255, 20, 147];[255,127,36]]/255;


fig = figure;
set(fig, 'DefaultAxesFontSize', 50);
set(fig, 'DefaultAxesFontWeight', 'bold');

set(fig, 'PaperSize', [7*2.5 6*2.5]);

date_str_list={'har/1222_213035_5322','openimage/1222_213332_35442','google_speech/1222_213948_48148','stackoverflow/1222_213529_50130','google_speech/1222_164841_19292'};

data_root_path='/mnt/home/lichenni/projects/PyramidFL/training/evals/logs/';

for ii=1:length(date_str_list)
    data_root = [data_root_path,date_str_list{ii},'/worker/'];

    error_path = [data_root, 'obs_client_distribution.mat'];
    a = load(error_path);
    error_matrix = struct2cell(a);

    client_size = error_matrix{3};
    client_size=cast(client_size,'double');
    if ii<5
        client_size=(client_size-min(client_size))/(max(client_size)-min(client_size));
    else
        client_size=client_size/max(client_size);
    end
    [F,X,Flo,Fup] = ecdf(client_size);
    plot(X,F,"-*",'LineWidth',8,'color',color_list(ii,:)); 
    hold on;
end
legend({['HARBox'],['OpenImage'], ['Google Speech'], ['Stackoverflow'], ['IID Distribution']}, 'FontSize', 40, 'Location', 'northwest','NumColumns',1);

xlabel('Normalized Data Size'); % x label
ylabel('CDF Across Clients'); % y label
title('')
xlim([1.3*1E-4, 1]);
set(gca, 'Xtick', [1E-3,1E-2,1E-1,1])
set(gca, 'XScale', 'log');
set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 640 * 2, 550 * 2]);
saveas(gcf, ['fig_clients_size.png'])
clf

for ii=1:length(date_str_list)-1
    data_root = [data_root_path,date_str_list{ii},'/aggregator/'];

    error_path = [data_root, 'obs_client_time.mat'];
    a = load(error_path)
    error_matrix = struct2cell(a);

    computation_list = error_matrix{4};
    computation_list=cast(computation_list,'double');
    computation_list=(computation_list-min(computation_list))/(max(computation_list)-min(computation_list));

    [F,X,Flo,Fup] = ecdf(computation_list);
    plot(X,F,"-*",'LineWidth',8,'color',color_list(ii,:)); 
    hold on;
end
legend({['HARBox'],['OpenImage'], ['Google Speech'], ['Stackoverflow']}, 'FontSize', 60, 'Location', 'northwest','NumColumns',1);

xlabel('Normalized Computation Speed'); % x label
ylabel('CDF Across Clients'); % y label
title('')
set(gca, 'XScale', 'log');
set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 640 * 2, 550 * 2]);
saveas(gcf, ['fig_computation_list.png'])
clf

for ii=1:length(date_str_list)-1
    data_root = [data_root_path,date_str_list{ii},'/aggregator/'];

    error_path = [data_root, 'obs_client_time.mat'];
    a = load(error_path);
    error_matrix = struct2cell(a);

    communication_list = error_matrix{5};
    plot_data=communication_list;
    plot_data=cast(plot_data,'double');
    plot_data=(plot_data-min(plot_data))/(max(plot_data)-min(plot_data));

    [F,X,Flo,Fup] = ecdf(plot_data);
    plot(X,F,"-*",'LineWidth',8,'color',color_list(ii,:)); 
    hold on;
end
legend({['HARBox'],['OpenImage'], ['Google Speech'], ['Stackoverflow']}, 'FontSize', 50, 'Location', 'northwest','NumColumns',1);

xlabel('Normalized Network Bandwidth'); % x label
ylabel('CDF Across Clients'); % y label
title('')
xlim([1*1E-4, 1]);
set(gca, 'Xtick', [1E-4,1E-3,1E-2,1E-1,1])
set(gca, 'XScale', 'log');
set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 640 * 2, 550 * 2]);
saveas(gcf, ['fig_communication_list.png'])
clf

date_str_list={'har/1222_213035_5322','openimage/1222_213332_35442','google_speech/1222_213948_48148','google_speech/1222_164841_19292'};

for ii=1:length(date_str_list)
    date_str_list
    data_root = [data_root_path,date_str_list{ii},'/worker/'];
    error_path = [data_root, 'obs_client_distribution.mat'];
    a = load(error_path)
    error_matrix = struct2cell(a);

    client_label = error_matrix{2};
    client_label=cast(client_label,'double');
    if ii<4
        client_label=(client_label-min(client_label))/(max(client_label)-min(client_label));
    else
        client_label=client_label/max(client_label);
    end
    [F,X,Flo,Fup] = ecdf(client_label);
    plot(X,F,"-*",'LineWidth',8,'color',color_list(ii,:)); 
    hold on;
end
legend({['HARBox'],['OpenImage'],['Google Speech'],['IID Distribution']}, 'FontSize', 40, 'Location', 'southeast','NumColumns',1);
xlabel('Normalized Data Labels Numbers'); % x label
ylabel('CDF Across Clients'); % y label
title('');
set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 640 * 2, 550 * 2]);
saveas(gcf, ['fig_clients_labels.png'])
clf

for ii=1:length(date_str_list)
    date_str_list{ii}
    data_root = [data_root_path,date_str_list{ii},'/worker/'];

    error_path = [data_root, 'obs_client_distribution.mat'];
    a = load(error_path);
    error_matrix = struct2cell(a);

    emd_distance = error_matrix{1};
    emd_distance=cast(emd_distance,'double');
    emd_distance=emd_distance;

    [F,X,Flo,Fup] = ecdf(emd_distance);
    plot(X,F,"-*",'LineWidth',8,'color',color_list(ii,:)); 
    hold on;
end
legend({['HARBox'],['OpenImage'],['Google Speech'],['IID Distribution']}, 'FontSize', 45, 'Location', 'southeast','NumColumns',1);

xlabel('Pairwise J-S Data Divergence'); % x label
ylabel('CDF Across Clients'); % y label
title('')
set(gcf, 'WindowStyle', 'normal', 'Position', [0, 0, 640 * 2, 550 * 2]);
saveas(gcf, ['fig_clients_divergence.png'])