

# https://www2.deloitte.com/content/dam/Deloitte/global/Documents/Financial-Services/gx-be-aers-fsi-credit-scoring.pdf

#---------------------------------
#  Thực hiện tiền xử lí số liệu
#---------------------------------

# Load dữ liệu: 

rm(list = ls())
library(tidyverse)
library(magrittr)

hmeq <- read.csv("D:/Teaching/data_science_banking/hmeq/hmeq.csv")

# Viết một số hàm xử lí số liệu thiếu và dán lại nhãn: 
thay_na_mean <- function(x) {
  tb <- mean(x, na.rm = TRUE)
  x[is.na(x)] <- tb
  return(x)
}


name_job <- function(x) {
  x %<>% as.character()
  ELSE <- TRUE
  quan_tam <- c("Mgr", "Office", "Other", "ProfExe", "Sales", "Self")
  case_when(!x %in% quan_tam ~ "Other", 
            ELSE ~ x)
}


name_reason <- function(x) {
  ELSE <- TRUE
  x %<>% as.character()
  case_when(!x %in% c("DebtCon", "HomeImp") ~ "Unknown", 
            ELSE ~ x)
}

label_rename <- function(x) {
  case_when(x == 1 ~ "BAD", 
            x == 0 ~ "GOOD")
}


my_scale01 <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}


# Xử lí số liệu thiếu và dán nhãn lại: 
df <- hmeq %>% 
  mutate_if(is.numeric, thay_na_mean) %>% 
  mutate_at("REASON", name_reason) %>% 
  mutate_at("JOB", name_job) %>% 
  mutate(BAD = label_rename(BAD)) %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate_if(is.numeric, my_scale01)

#-----------------------------------------------
#  Chuẩn bị dữ liệu và chế độ Cross Validation
#-----------------------------------------------

library(caret)
set.seed(1)
id <- createDataPartition(y = df$BAD, p = 0.5, list = FALSE)

train <- df[id, ]
test <- df[-id, ]

# Thiết lập môi trường tinh chỉnh tham số và cross - validation: 

set.seed(1)
train.control <- trainControl(method = "repeatedcv", 
                              number = 5,
                              repeats = 5, 
                              classProbs = TRUE,
                              allowParallel = TRUE, 
                              summaryFunction = multiClassSummary)

# Thiết lập chế độ tính toán song song. Bước này có thể rút 
# gọn thời gian tính toán - chạy mô hình chỉ còn 1/3: 

library(doParallel)
n_cores <- detectCores()

# Sử dụng 7 nhân: 
registerDoParallel(cores = n_cores - 2)

# Huấn luyện RF: 
set.seed(1)
my_logit <- train(BAD ~., 
                  data = train, 
                  method = "glm", 
                  family = "binomial", 
                  trControl = train.control, 
                  tuneLength = 5)


# Viết cái hàm (kiểu 1) tính toán trung bình cho các tiêu 
# chí đánh giá chất lượng phân loại của mô hình với cách 
# thức chọn 1000 mẫu quan sát từ testing data với 100 lần: 

eval_fun1 <- function(thre, model_selected) {
  my_df <- data.frame()
  for (i in 1:100) {
    set.seed(i)
    id <- createDataPartition(y = test$BAD, p = 1000 / nrow(test), list = FALSE)
    test_df <- test[id, ]
    
    du_bao <- predict(model_selected, test_df, type = "prob") %>% 
      pull(BAD)
    
    du_bao <- case_when(du_bao >= thre ~ "BAD", 
                        du_bao < thre ~ "GOOD")
    
    cm <- confusionMatrix(test_df$BAD, du_bao %>% as.factor())
    
    bg_gg <- cm$table %>% 
      as.vector() %>% 
      matrix(ncol = 4) %>% 
      as.data.frame()
    
    names(bg_gg) <- c("BB", "GB", "BG", "GG")
    
    
    kq <- c(cm$overall, cm$byClass) 
    ten <- kq %>% as.data.frame() %>% row.names()
    
    kq %>% 
      as.vector() %>% 
      matrix(ncol = 18) %>% 
      as.data.frame() -> all_df
    
    names(all_df) <- ten
    all_df <- bind_cols(all_df, bg_gg)
    my_df <- bind_rows(my_df, all_df)
    
  }
  return(my_df)
}


# Viết hàm tính toán khả năng phân loại của mô hình dựa 
# trên một loạt ngưỡng được lựa chọn trước: 

my_results_from_thres_range <- function(low_thres, up_thres, by, model_selected) {
  my_range <- seq(low_thres, up_thres, by = by)
  n <- length(my_range)
  all_df <- data.frame()
  
  for (i in 1:n) {
    df <- eval_fun1(my_range[i], model_selected = model_selected)
    df %<>% mutate(Threshold = my_range[i])
    all_df <- bind_rows(all_df, df)
  }
  return(all_df)
}



# Sử dụng hàm: 

my_results_from_thres_range(0.1, 0.8, 0.1, my_logit) ->> logit_results

# Đổi tên: 
names(logit_results) <- names(logit_results) %>% str_replace_all(" ", "")


# Hình ảnh hóa sự biển đổi của một số tiêu chí đánh giá
# khả năng phân loại của mô hình khi ngưỡng thay đổi: 

theme_set(theme_minimal())
logit_results %>% 
  group_by(Threshold) %>% 
  summarise_each(funs(median), Accuracy, NegPredValue, PosPredValue) %>% 
  ungroup() %>% 
  gather(Metric, b, -Threshold) %>% 
  ggplot(aes(Threshold, b, color = Metric)) + 
  geom_line() + 
  geom_point() + 
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1))


#-------------------------------------------------------------
#   Đánh giá lợi nhuận dựa trên mô phỏng khi ngưỡng thay đổi
#-------------------------------------------------------------


logit_results %>% 
  group_by(Threshold) %>% 
  summarise_each(funs(sum), GG, BG) ->> gg_bg_LG_Model


# Viết hàm mô phỏng lợi nhuận với 1000 lần mô phỏng: 

profit_simu <- function(data_from_model, dong, rate) {
  prof <- c()
  for (j in 1:1000) {
    
    vay_tot <- data_from_model[dong, 2] %>% as.numeric()
    vay_xau <- data_from_model[dong, 3] %>% as.numeric()
    
    so_tien_cho_vay_tot <- sample(hmeq$LOAN, vay_tot, replace = TRUE)
    so_tien_cho_vay_xau <- sample(hmeq$LOAN, vay_xau, replace = TRUE)
    
    loi_nhuan <- sum(rate*so_tien_cho_vay_tot) - sum(so_tien_cho_vay_xau)
    prof <- c(prof, loi_nhuan)
  }
  
  data.frame(Profit = prof, Threshold = data_from_model[dong, 1] %>% as.vector()) %>% 
    return()
  
}


# Lợi nhuận mô phỏng khi sử dụng mô hình Random Forest: 

logit_profit <- bind_rows(profit_simu(gg_bg_LG_Model, 1, 0.1), 
                          profit_simu(gg_bg_LG_Model, 2, 0.1), 
                          profit_simu(gg_bg_LG_Model, 3, 0.1), 
                          profit_simu(gg_bg_LG_Model, 4, 0.1), 
                          profit_simu(gg_bg_LG_Model, 5, 0.1), 
                          profit_simu(gg_bg_LG_Model, 6, 0.1), 
                          profit_simu(gg_bg_LG_Model, 7, 0.1), 
                          profit_simu(gg_bg_LG_Model, 8, 0.1))



# Sự biến đổi của lợi nhuận trung bình khi ngưỡng thay đổi: 
logit_profit %>% 
  mutate(Threshold = as.factor(Threshold)) %>% 
  ggplot(aes(Threshold, Profit / 1000)) + 
  geom_boxplot()


# với ngưỡng mặc định 0.5 thì ma trận nhầm lẫn là: 

confusionMatrix(test$BAD, predict(my_logit, test))


# Viết hàm tính ma trận nhầm lẫn với ngưỡng được chọn: 

my_cm_thres <- function(model_selected, thre, test_df) {
  du_bao <- predict(model_selected, test_df, type = "prob") %>% 
    pull(BAD)
  
  du_bao <- case_when(du_bao >= thre ~ "BAD", 
                      du_bao < thre ~ "GOOD")
  
  cm <- confusionMatrix(test_df$BAD, du_bao %>% as.factor())
  
  return(cm)
}

# Kết quả với ngưỡng mặc định như trên có thể tái lập
# bằng cách sử dụng hàm đã viết: 

my_cm_thres(my_logit, 0.5, test)

# Con tại ngưỡng 0.1: 
my_cm_thres(my_logit, 0.1, test)
























