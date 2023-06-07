function result(true_value,predict_value)
rmse=sqrt(mean((true_value-predict_value).^2));
disp(['root mean square error(RMSE)��',num2str(rmse)])
mae=mean(abs(true_value-predict_value));
disp(['mean absolute error��MAE����',num2str(mae)])

mape=mean(abs(true_value-predict_value)/true_value);
disp(['mean relative percent error��MAPE����',num2str(mape*100),'%'])
