function OH = OHspecial_transform(df)
    % Be tolerant of missing table helpers (Octave) without changing math.
    [names, data] = get_names_and_data(df);
    size(data)
    for j = 1:length(names)
        temp = unique(data(:,j));
        temp = temp(temp>=0);
        var = zeros([size(data,1),length(temp)]);
        
        for k = 1:size(data,1)
            for l = 1:length(temp)
                
                if data(k,j) == temp(l)
                    var(k,l) = 1;
                end
            end
        end
        idx = find(isnan(data(:,j))==1);
        var(idx,:) = NaN;

        if j > 1
            clear label
        end
        for l = 1:length(temp)
            label{l} = sprintf('%s_%s', names{j}, num2str(temp(l)));
        end

        if j == 1
            labels = label;
            OH = var;
            if size(var,2) <= 2
                OH = var(:,end);
                labels = labels(end);
            end
        elseif j > 1
            if size(var,2) <= 2
                var = var(:,end);
                label = label(end);
            end
            labels = [labels,label];
            OH = [OH,var];
        end
    end
    OH = wrap_table_like(OH, labels);
end

function [names, data] = get_names_and_data(df)
    % Prefer table APIs when present; otherwise fall back to struct/numeric inputs.
    if exist('istable', 'file') == 2 && istable(df)
        names = df.Properties.VariableNames;
        data = table2array(df);
        return
    end

    % Handle MATLAB table objects even if istable is missing (Octave).
    if isstruct(df) && isfield(df, 'Properties') && isfield(df.Properties, 'VariableNames')
        names = df.Properties.VariableNames;
        if exist('table2array', 'file') == 2
            data = table2array(df);
        else
            data = df{:,:};
        end
        return
    end

    % Compat path: struct produced by readtable_compat.
    if isstruct(df) && isfield(df, 'data')
        data = df.data;
        if isfield(df, 'names')
            names = df.names;
        else
            names = default_names(size(data, 2));
        end
        return
    end

    % Plain numeric matrix.
    if isnumeric(df)
        data = df;
        names = default_names(size(data, 2));
        return
    end

    error('Unsupported input type for OHspecial_transform');
end

function names = default_names(ncols)
    names = cell(1, ncols);
    for c = 1:ncols
        names{c} = sprintf('col%d', c);
    end
end

function T = wrap_table_like(OH, labels)
    if exist('array2table', 'file') == 2
        T = array2table(OH);
        T.Properties.VariableNames = labels;
    else
        % Return a struct with data and names when tables are unavailable.
        T = struct('data', OH, 'names', labels);
    end
end



