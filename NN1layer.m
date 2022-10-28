input = [0 0 1; 0 1 1; 1 0 1; 1 1 1];
ref_output = [0;0; 1; 1];
weight = 2*rand(1,3)-1; % normalised weights

for epoch = 1:20000
    weight = SGD(weight, input , ref_output);
end
save('trained_NN.mat')


function weight = SGD(weight, input, ref_output)
alpha = 0.9;
N = 3;
for k = 1:N
    input_tp = input(k,:)';
    d = ref_output;
    weighted_sum = weight*input_tp;
    weighted_sum = dlarray(weighted_sum);
    output = sigmoid(weighted_sum);
    error = d - output;
    delta = output*(1-output)*error;
    dweight = alpha*(delta*input_tp');
    for i = 1:3
        weight(i) = weight(i)+ dweight(i);
    end
end
end