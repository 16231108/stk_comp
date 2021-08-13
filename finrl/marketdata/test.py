import baostock as bs
import pandas as pd

# 登陆系统
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

# 获取行业分类数据
rs = bs.query_stock_industry()
# rs = bs.query_stock_basic(code_name="浦发银行")
print('query_stock_industry error_code:'+rs.error_code)
print('query_stock_industry respond  error_msg:'+rs.error_msg)

# 打印结果集
industry_list = []
lxc_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    temp = rs.get_row_data()
    lxc_temp = temp
    if(temp[3]=="食品饮料"):
        temp_rs = bs.query_stock_basic(code=temp[1])
        temp_list =[]
        while (temp_rs.error_code == '0') & temp_rs.next():
            # 获取一条记录，将记录合并在一起
            temp_temp = temp_rs.get_row_data()
            temp_list.append(temp_temp[2])
        #print("temp_list is:",temp_list)
        temp.append(temp_list[0])
        lxc_list.append(temp)
    industry_list.append(lxc_temp)
list_columns = rs.fields
list_columns.append("上市时间")
result = pd.DataFrame(industry_list, columns=list_columns)
lxc_result = pd.DataFrame(lxc_list, columns=list_columns)
# 结果集输出到csv文件
result.to_csv("D:/stock_industry.csv", encoding="gbk", index=False)
lxc_result.to_csv("D:/食品饮料_industry.csv", encoding="gbk", index=False)
print(lxc_result)

# 登出系统
bs.logout()