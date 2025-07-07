loss1 = load("loss1_NoStag_LF1_Medium_dividedByPsquared.dat");
loss2 = load("loss2_NoStag_LF1_Medium_dividedByPsquared.dat");

figure;
plot(loss1, loss2, '.')
% plot(loss1, '.')
% hold on;
% plot(loss2, '+')
% legend("Loss1", "Loss2")