# 🚀 系统设计文档 - 货源分配前后端平台

我在本地已经实现一个货源分配算法， 我想实现一个前端的工作，
constraint_manager是约束验证的文件
data_loader是数据加载的逻辑
linear_programming是相关算法的逻辑
这几个文件保留的基础上，根据他们的逻辑帮我重新根据fastapi设计一个后端服务在backend目录下，依这三个文件的逻辑设计一个api
前端实现在frontend 文件下



## 📌 一、系统目标



构建一个用户友好的 Web 平台，支持：
- 上传 Excel 数据；
- 选择约束条件和目标函数；
- 提交计算任务；
- 展示结果（分轮分配表、约束校验）；
- 支持导出结果或结果图表。

## 🧱 二、技术栈

| 层级 | 技术/框架            | 说明                   |
| ---- | -------------------- | ---------------------- |
| 前端 | React+Ant Design Pro | UI 框架                |
| 后端 | FastAPI              | API服务                |
| 数据 | pandas + pulp        | 数据处理与线性规划计算 |

## 🖥 三、系统模块划分
主要分为三个tab
### 1. 首页 - Excel 导入与计算配置

#### 📄 主要功能
- 上传 Excel 文件（Sheet1/Sheet2）  导入的样式在huoyuanfenpei.xlsx里
- 动态解析轮次
- 配置约束（solve里）  相关的参数提出出来也做配置 比如投放总量的值可以自己选  单箱均价的上下限
        # 添加基本约束
        self.add_demand_constraints(self.model, self.variables)  # 约束1：需求满足
        self.add_volume_constraints(self.model, self.variables)  # 约束3：投放总量
        self.add_average_price_constraints(self.model, self.variables)  # 约束2：单箱均价
        
        # 添加分配策略约束
        self.add_demand_split_constraints(self.model, self.variables)  # 约束6：需求量分轮（仅非existing_allocations）
        self.add_demand_based_constraints(self.model, self.variables)  # 约束8：按需优先（软约束）
        self.add_price_based_constraints(self.model, self.variables)  # 约束9：按价比例
        
        # 添加C类烟约束
        self.add_c_type_constraints(self.model, self.variables)  # 约束10.1-10.2
        self._add_c_subtype_constraints(self.model, self.variables)  # 约束10.3-10.5
        
        # 添加均衡约束
        self.add_balance_constraints(self.model, self.variables)  # 约束11：分配均衡（软约束）
- 配置目标函数（create_objective_function里）
- 点击按钮提交计算任务

#### 🧩 前端设计

| 元素          | 说明                 |
| ------------- | -------------------- |
| Upload 组件   | 上传 `.xlsx` 文件    |
| Form + Select | 用于设置约束选项开关 |
| Form + Switch | 目标函数配置选项     |
| Button        | 开始计算             |
| Spin/Loading  | 计算中状态提示       |
成功后提示去分配明细查看 & 导出页面查看


### 2. 分配明细查看 & 导出
oupput4就是linear_programing 生成结果中的result['allocation_matrix']
#### 📄 主要功能
- 全量导出最终计算结果 导出的样式在output4.xlsx里  
- 展示分配结果，样式output4.xlsx里
- 分配状态 是否有最优解 求解时间等可以看情况展示  

#### 🧩 前端设计

| 元素   | 说明                |
| ------ | ------------------- |
| Table  |                     |
| Button | 导出为 Excel 或 CSV |

---



### 3. 计算结果页面

#### 📄 主要功能
- 展示按品规的投放汇总信息  样式按照output4.xlsx
- 展示是否满足约束（可视化/表格） 根据constraint_manager验证的返回信息

#### 🧩 前端设计

| 元素  | 说明                   |
| ----- | ---------------------- |
| Table | 按品规的投放汇总信息表 |
|       | 展示各类约束是否满足   |

---

## 🧩 四、后端接口设计（FastAPI）

### ✅ 接口列表

| 接口路径        | 方法 | 说明                       |
| --------------- | ---- | -------------------------- |
| `/upload/`      | POST | 上传 Excel，返回解析信息   |
| `/solve/`       | POST | 传入计算配置，返回计算结果 |
| `/result/`      | GET  | 获取上一次计算的详细结果   |
| `/export/`      | GET  | 导出当前结果为 Excel/CSV   |
| `/constraints/` | GET  | 获取当前约束的验证结果     |


