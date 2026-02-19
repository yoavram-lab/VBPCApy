function OH = OHspecial_transform(df)
    names = string(df.Properties.VariableNames);
    data = table2array(df);
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
            label(l) = string(strcat(names(j),'_',num2str(temp(l))));
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
    OH = array2table(OH);
    OH.Properties.VariableNames = labels;
end



