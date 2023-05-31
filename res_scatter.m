function [] = res_scatter(predY, testY)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
pos_index = find(testY);
pos_index = pos_index(:, 1);
neg_index = find(1-testY);
neg_index = neg_index(:, 1);

scatter(pos_index, predY(pos_index), 'r')
hold on 
scatter(neg_index, predY(neg_index), 'b')
xlabel('sample\_index')
ylabel('predication')

end

