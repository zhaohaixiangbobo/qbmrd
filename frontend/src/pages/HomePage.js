import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  Upload,
  Button,
  Form,
  Switch,
  InputNumber,
  Divider,
  message,
  Spin,
  Alert,
  Row,
  Col,
  Typography,
  Space,
  Tag,
  Descriptions,
  Tooltip
} from 'antd';
import {
  InboxOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  QuestionCircleOutlined,
  DownloadOutlined
} from '@ant-design/icons';
import { uploadFile, solveAllocation } from '../services/api';

const { Dragger } = Upload;
const { Title, Text } = Typography;

/**
 * 首页组件 - 数据导入与计算配置
 * 支持状态持久化，保留上传文件和约束配置状态
 */
const HomePage = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();
  const [uploadData, setUploadData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [solving, setSolving] = useState(false);
  const [solveResult, setSolveResult] = useState(null);

  // 状态持久化的键名
  const STORAGE_KEYS = {
    UPLOAD_DATA: 'homepage_upload_data',
    FORM_VALUES: 'homepage_form_values',
    SOLVE_RESULT: 'homepage_solve_result'
  };

  /**
   * 从localStorage加载保存的状态
   */
  const loadPersistedState = useCallback(() => {
    try {
      // 加载上传数据
      const savedUploadData = localStorage.getItem(STORAGE_KEYS.UPLOAD_DATA);
      if (savedUploadData) {
        const parsedUploadData = JSON.parse(savedUploadData);
        setUploadData(parsedUploadData);
      }

      // 加载计算结果
      const savedSolveResult = localStorage.getItem(STORAGE_KEYS.SOLVE_RESULT);
      if (savedSolveResult) {
        const parsedSolveResult = JSON.parse(savedSolveResult);
        setSolveResult(parsedSolveResult);
      }

      // 加载表单值
      const savedFormValues = localStorage.getItem(STORAGE_KEYS.FORM_VALUES);
      if (savedFormValues) {
        const parsedFormValues = JSON.parse(savedFormValues);
        // 延迟设置表单值，确保表单已完全渲染
        setTimeout(() => {
          form.setFieldsValue(parsedFormValues);
        }, 100);
      }
    } catch (error) {
      console.error('加载保存的状态失败:', error);
      // 如果加载失败，清除可能损坏的数据
      Object.values(STORAGE_KEYS).forEach(key => {
        localStorage.removeItem(key);
      });
    }
  }, [form]);

  /**
   * 保存状态到localStorage
   */
  const saveStateToStorage = (key, data) => {
    try {
      localStorage.setItem(key, JSON.stringify(data));
    } catch (error) {
      console.error('保存状态失败:', error);
    }
  };

  /**
   * 清除所有保存的状态
   */
  const clearPersistedState = () => {
    Object.values(STORAGE_KEYS).forEach(key => {
      localStorage.removeItem(key);
    });
  };

  // 组件挂载时加载保存的状态
  useEffect(() => {
    loadPersistedState();
  }, [loadPersistedState]);

  // 设置表单初始值
  useEffect(() => {
    if (uploadData) {
      // 使用setTimeout确保表单完全渲染后再设置初始值
      setTimeout(() => {
        const initialValues = {
          enable_demand_constraints: true,
          enable_volume_constraints: true,
          enable_price_constraints: true,
          enable_c_type_constraints: true,
          enable_balance_constraints: true,
          enable_demand_split_constraints: true,
          enable_demand_based_constraints: true,
          enable_price_based_constraints: true,
          volume_tolerance: 0.005,
          price_based_ratio: 0.3,
          c_type_ratio: 0.4,
          c_type_volume_limit: 4900,
          chang_type_ratio: 0.2,
          chang_type_volume_limit: 1000,
          xi_type_ratio: 0.6,
          xi_type_volume_limit: 3000,
          enable_maximize_allocation: true,
          maximize_allocation_weight: 1000.0,
          enable_round_balance: false,
          round_balance_weight: 800.0,
          enable_round_variance: false,
          round_variance_weight: 400.0,
          enable_product_balance: false,
          product_balance_weight: 100.0,
          enable_smooth_transition: true,
          smooth_transition_weight: 300.0
        };
        
        // 添加轮次约束的初始值
        if (uploadData.rounds && uploadData.round_constraints) {
          uploadData.rounds.forEach(round => {
            if (uploadData.round_constraints[round]) {
              initialValues[`price_upper_limits_${round}`] = uploadData.round_constraints[round].upper_price_limit;
              initialValues[`price_lower_limits_${round}`] = uploadData.round_constraints[round].lower_price_limit;
              initialValues[`volume_limits_${round}`] = uploadData.round_constraints[round].total_quantity;
            }
          });
        }
        
        // 检查是否有保存的表单值
        const savedFormValues = localStorage.getItem(STORAGE_KEYS.FORM_VALUES);
        const formValues = savedFormValues ? 
          { ...initialValues, ...JSON.parse(savedFormValues) } : 
          initialValues;
        
        // 设置表单值
        form.setFieldsValue(formValues);
        
        // 强制触发表单更新
        form.validateFields().catch(() => {});
      }, 100);
    }
  }, [form, uploadData]);

  /**
   * 处理表单值变化，自动保存到localStorage
   */
  const handleFormValuesChange = useCallback((changedValues, allValues) => {
    saveStateToStorage(STORAGE_KEYS.FORM_VALUES, allValues);
  }, [STORAGE_KEYS.FORM_VALUES]);

  // 下载模板文件
  const handleDownloadTemplate = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/download-template');
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'huoyuanfenpei.xlsx';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        message.success('模板文件下载成功！');
      } else {
        message.error('模板文件下载失败');
      }
    } catch (error) {
      console.error('下载模板失败:', error);
      message.error('模板文件下载失败');
    }
  };

  /**
   * 文件上传处理
   */
  const handleUpload = async (file) => {
    setLoading(true);
    try {
      const response = await uploadFile(file);
      if (response.data.success) {
        const newUploadData = response.data.data;
        
        // 清除之前保存的表单值，确保使用新上传文件的初始值
        localStorage.removeItem(STORAGE_KEYS.FORM_VALUES);
        
        // 更新上传数据并保存到localStorage
        setUploadData(newUploadData);
        saveStateToStorage(STORAGE_KEYS.UPLOAD_DATA, newUploadData);
        
        message.success('文件上传成功！');
      } else {
        message.error(response.data.message || '文件上传失败');
      }
    } catch (error) {
      console.error('上传失败:', error);
      message.error('文件上传失败，请检查文件格式');
    } finally {
      setLoading(false);
    }
    return false; // 阻止默认上传行为
  };

  /**
   * 开始计算
   */
  const handleSolve = async () => {
    try {
      const values = await form.validateFields();
      setSolving(true);
      
      // 处理轮次约束数据
      const price_upper_limits = {};
      const price_lower_limits = {};
      const volume_limits = {};
      
      if (uploadData && uploadData.rounds) {
        uploadData.rounds.forEach(round => {
          const upperLimit = values[`price_upper_limits_${round}`];
          const lowerLimit = values[`price_lower_limits_${round}`];
          const volumeLimit = values[`volume_limits_${round}`];
          
          if (upperLimit !== undefined && upperLimit !== null) {
            price_upper_limits[round] = upperLimit;
          }
          if (lowerLimit !== undefined && lowerLimit !== null) {
            price_lower_limits[round] = lowerLimit;
          }
          if (volumeLimit !== undefined && volumeLimit !== null) {
            volume_limits[round] = volumeLimit;
          }
        });
      }
      
      const config = {
        constraints: {
          enable_demand_constraints: values.enable_demand_constraints !== undefined ? values.enable_demand_constraints : true,
          enable_volume_constraints: values.enable_volume_constraints !== undefined ? values.enable_volume_constraints : true,
          enable_price_constraints: values.enable_price_constraints !== undefined ? values.enable_price_constraints : true,
          enable_c_type_constraints: values.enable_c_type_constraints !== undefined ? values.enable_c_type_constraints : true,
          enable_balance_constraints: values.enable_balance_constraints !== undefined ? values.enable_balance_constraints : true,
          enable_demand_split_constraints: values.enable_demand_split_constraints !== undefined ? values.enable_demand_split_constraints : true,
          enable_demand_based_constraints: values.enable_demand_based_constraints !== undefined ? values.enable_demand_based_constraints : true,
          enable_price_based_constraints: values.enable_price_based_constraints !== undefined ? values.enable_price_based_constraints : true,
          // 根据开关状态设置约束参数，禁用时设为null或不传递
          volume_tolerance: values.enable_volume_constraints ? 
            (values.volume_tolerance ?? 0.005) : 0.005,
          // 轮次约束参数
          price_upper_limits: Object.keys(price_upper_limits).length > 0 ? price_upper_limits : price_upper_limits,
          price_lower_limits: Object.keys(price_lower_limits).length > 0 ? price_lower_limits : price_lower_limits,
          volume_limits: Object.keys(volume_limits).length > 0 ? volume_limits : volume_limits,
          // 按价比例约束参数
          price_based_ratio: values.enable_price_based_constraints ? 
            (values.price_based_ratio ?? 0.3) : 0.3,
          // C类烟约束参数
          c_type_ratio: values.enable_c_type_constraints ? 
            (values.c_type_ratio ?? 0.4) : 0.4,
          c_type_volume_limit: values.enable_c_type_constraints ? 
            (values.c_type_volume_limit ?? 4900) : 4900,
          chang_type_ratio: values.enable_c_type_constraints ? 
            (values.chang_type_ratio ?? 0.2) : 0.2,
          chang_type_volume_limit: values.enable_c_type_constraints ? 
            (values.chang_type_volume_limit ?? 1000) : 1000,
          xi_type_ratio: values.enable_c_type_constraints ? 
            (values.xi_type_ratio ?? 0.6) : 0.6,
          xi_type_volume_limit: values.enable_c_type_constraints ? 
            (values.xi_type_volume_limit ?? 3000) : 3000
        },
        objective: {
          // 根据开关状态设置权重，禁用时设为0
          maximize_allocation_weight: values.enable_maximize_allocation ? 
            (values.maximize_allocation_weight ?? 1000.0) : 0.0,
          round_balance_weight: values.enable_round_balance ? 
            (values.round_balance_weight ?? 800.0) : 0.0,
          round_variance_weight: values.enable_round_variance ? 
            (values.round_variance_weight ?? 400.0) : 0.0,
          product_balance_weight: values.enable_product_balance ? 
            (values.product_balance_weight ?? 100.0) : 0.0,
          smooth_transition_weight: values.enable_smooth_transition ? 
            (values.smooth_transition_weight ?? 300.0) : 0.0
        }
      };

      const response = await solveAllocation(config);
      if (response.data.success) {
        const newSolveResult = response.data.data;
        setSolveResult(newSolveResult);
        // 保存计算结果到localStorage
        saveStateToStorage(STORAGE_KEYS.SOLVE_RESULT, newSolveResult);
        message.success('计算完成！');
      } else {
        message.error(response.data.message || '计算失败');
      }
    } catch (error) {
      console.error('计算失败:', error);
      message.error('计算失败，请检查配置参数');
    } finally {
      setSolving(false);
    }
  };

  return (
    <div>
      <Title level={2}>
        <InboxOutlined style={{ marginRight: 8 }} />
        数据导入与计算配置
      </Title>
      
      {/* 文件上传区域 */}
      <Card 
        title="1. Excel 文件上传" 
        style={{ marginBottom: 24 }}
        extra={
          <Space>
            <Button 
              icon={<DownloadOutlined />}
              onClick={handleDownloadTemplate}
              size="small"
            >
              下载模板
            </Button>
            {uploadData && (
              <>
                <Tag color="green">已上传文件</Tag>
                <Button 
                  size="small" 
                  onClick={() => {
                    setUploadData(null);
                    setSolveResult(null);
                    localStorage.removeItem(STORAGE_KEYS.UPLOAD_DATA);
                    localStorage.removeItem(STORAGE_KEYS.SOLVE_RESULT);
                    form.resetFields();
                    message.success('已清除上传文件，可重新上传');
                  }}
                >
                  重新上传
                </Button>
              </>
            )}
          </Space>
        }
      >
        <Spin spinning={loading}>
          {!uploadData ? (
            <Dragger
              name="file"
              multiple={false}
              accept=".xlsx,.xls"
              beforeUpload={handleUpload}
              showUploadList={false}
              style={{ marginBottom: 16 }}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
              <p className="ant-upload-hint">
                支持 .xlsx 和 .xls 格式的Excel文件
              </p>
            </Dragger>
          ) : (
            <Alert
              message="文件上传成功"
              description={
                <Descriptions size="small" column={2}>
                  <Descriptions.Item label="产品数量">{uploadData.total_products}</Descriptions.Item>
                  <Descriptions.Item label="轮次数量">{uploadData.rounds.length}</Descriptions.Item>
                  <Descriptions.Item label="轮次列表">
                    <Space>
                      {uploadData.rounds.map(round => (
                        <Tag key={round} color="blue">{round}</Tag>
                      ))}
                    </Space>
                  </Descriptions.Item>
                  <Descriptions.Item label="上传时间">{new Date(uploadData.upload_time).toLocaleString()}</Descriptions.Item>
                </Descriptions>
              }
              type="success"
              showIcon
              style={{ marginBottom: 16 }}
            />
          )}
          
          {/* 状态持久化提示 */}
          {uploadData && (
            <Alert
              message="状态已保存"
              description="您的文件上传状态和配置参数已自动保存，切换页面后再回来时会自动恢复。"
              type="info"
              showIcon
              closable
            />
          )}
        </Spin>
      </Card>

      {/* 约束配置 */}
      {uploadData && (
        <Card 
          title="2. 约束条件配置" 
          style={{ marginBottom: 24 }}
          extra={
            <Space>
              <Button 
                size="small" 
                onClick={() => {
                  form.resetFields();
                  localStorage.removeItem(STORAGE_KEYS.FORM_VALUES);
                  message.success('已重置为默认配置');
                }}
              >
                重置配置
              </Button>
              <Button 
                size="small" 
                danger
                onClick={() => {
                  clearPersistedState();
                  setUploadData(null);
                  setSolveResult(null);
                  form.resetFields();
                  message.success('已清除所有保存的状态');
                }}
              >
                清除所有状态
              </Button>
            </Space>
          }
        >
          <Form
            form={form}
            layout="vertical"
            key="constraint-form"
            onValuesChange={handleFormValuesChange}
            initialValues={{
              // 基础约束开关默认值
              enable_demand_constraints: true,
              enable_volume_constraints: true,
              enable_price_constraints: true,
              enable_c_type_constraints: true,
              enable_balance_constraints: true,
              enable_demand_split_constraints: true,
              enable_demand_based_constraints: true,
              enable_price_based_constraints: true,
              // 目标函数开关默认值
              enable_maximize_allocation: true,
              enable_round_balance: true,
              enable_round_variance: true,
              enable_product_balance: true,
              enable_smooth_transition: true,
              // 约束参数默认值
              volume_tolerance: 0.005,
              price_based_ratio: 0.3,
              c_type_ratio: 0.4,
              c_type_volume_limit: 4900,
              chang_type_ratio: 0.2,
              chang_type_volume_limit: 1000,
              xi_type_ratio: 0.6,
              xi_type_volume_limit: 3000,
              // 目标函数权重默认值
              maximize_allocation_weight: 1000.0,
              round_balance_weight: 800.0,
              round_variance_weight: 400.0,
              product_balance_weight: 100.0,
              smooth_transition_weight: 300.0
            }}
          >
            <div className="config-section">
              <Title level={4}>
                <SettingOutlined style={{ marginRight: 8 }} />
                基础约束
              </Title>
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item name="enable_demand_constraints" valuePropName="checked" label="需求满足约束">
                    <Switch checkedChildren="启用" unCheckedChildren="禁用" defaultChecked={true} />
                    {/* <Text style={{ marginLeft: 8 }}>需求满足约束</Text> */}
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item name="enable_volume_constraints" valuePropName="checked" label="投放总量约束">
                    <Switch checkedChildren="启用" unCheckedChildren="禁用" defaultChecked={true} />
                    {/* <Text style={{ marginLeft: 8 }}>投放总量约束</Text> */}
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item name="enable_price_constraints" valuePropName="checked" label="单箱均价约束">
                    <Switch checkedChildren="启用" unCheckedChildren="禁用" defaultChecked={true}/>
                    {/* <Text style={{ marginLeft: 8 }}>单箱均价约束</Text> */}
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item name="enable_c_type_constraints" valuePropName="checked" label="C类烟约束">
                    <Switch checkedChildren="启用" unCheckedChildren="禁用" defaultChecked={true}  />
                    {/* <Text style={{ marginLeft: 8 }}>C类烟约束</Text> */}
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item name="enable_balance_constraints" valuePropName="checked" label="分配均衡约束">
                    <Switch checkedChildren="启用" unCheckedChildren="禁用" defaultChecked={true} />
                    {/* <Text style={{ marginLeft: 8 }}>分配均衡约束</Text> */}
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item name="enable_demand_split_constraints" valuePropName="checked" label="需求量集中约束">
                    <Switch checkedChildren="启用" unCheckedChildren="禁用" defaultChecked={true} />
                    {/* <Text style={{ marginLeft: 8 }}>需求量集中约束</Text> */}
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item name="enable_demand_based_constraints" valuePropName="checked" label="按需优先约束">
                    <Switch checkedChildren="启用" unCheckedChildren="禁用" defaultChecked={true} />
                    {/* <Text style={{ marginLeft: 8 }}>按需优先约束</Text> */}
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item name="enable_price_based_constraints" valuePropName="checked" label="按价比例约束">
                    <Switch checkedChildren="启用" unCheckedChildren="禁用" defaultChecked={true} />
                    {/* <Text style={{ marginLeft: 8 }}>按价比例约束</Text> */}
                  </Form.Item>
                </Col>
              </Row>
              
              {/* 约束配置参数 */}
              <Title level={5} style={{ marginTop: 16, marginBottom: 12 }}>
                <SettingOutlined style={{ marginRight: 8 }} />
                约束配置参数
              </Title>
              
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item dependencies={['enable_volume_constraints']} noStyle>
                    {({ getFieldValue }) => {
                      const enabled = getFieldValue('enable_volume_constraints');
                      return (
                        <Form.Item 
                          name="volume_tolerance" 
                          label="投放总量容差 (±%)"
                          tooltip="投放总量允许的上下浮动比例"
                        >
                          <InputNumber
                            min={0}
                            max={0.1}
                            step={0.001}
                            formatter={value => `${(value * 100).toFixed(1)}%`}
                            parser={value => value.replace('%', '') / 100}
                            style={{ width: '100%' }}
                            disabled={!enabled}
                          />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item dependencies={['enable_price_based_constraints']} noStyle>
                    {({ getFieldValue }) => {
                      const enabled = getFieldValue('enable_price_based_constraints');
                      return (
                        <Form.Item 
                          name="price_based_ratio" 
                          label="按价比例 (%)"
                          tooltip="按价品规在各轮中需占品规总数的最低比例"
                        >
                          <InputNumber
                            min={0}
                            max={1}
                            step={0.05}
                            formatter={value => `${(value * 100).toFixed(0)}%`}
                            parser={value => value.replace('%', '') / 100}
                            style={{ width: '100%' }}
                            placeholder="30%"
                            disabled={!enabled}
                          />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item dependencies={['enable_c_type_constraints']} noStyle>
                    {({ getFieldValue }) => {
                      const enabled = getFieldValue('enable_c_type_constraints');
                      return (
                        <Form.Item 
                          name="c_type_ratio" 
                          label="C类烟比例 (%)"
                          tooltip="C类烟在各轮中占总量的最大比例"
                        >
                          <InputNumber
                            min={0}
                            max={1}
                            step={0.05}
                            formatter={value => `${(value * 100).toFixed(0)}%`}
                            parser={value => value.replace('%', '') / 100}
                            style={{ width: '100%' }}
                            placeholder="40%"
                            disabled={!enabled}
                          />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>
                </Col>
              </Row>
              
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item dependencies={['enable_c_type_constraints']} noStyle>
                    {({ getFieldValue }) => {
                      const enabled = getFieldValue('enable_c_type_constraints');
                      return (
                        <Form.Item 
                          name="c_type_volume_limit" 
                          label="C类烟量限制 (箱)"
                          tooltip="每轮C类烟总量的最大限制"
                        >
                          <InputNumber
                            min={0}
                            step={100}
                            style={{ width: '100%' }}
                            placeholder="4900"
                            disabled={!enabled}
                          />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item dependencies={['enable_c_type_constraints']} noStyle>
                    {({ getFieldValue }) => {
                      const enabled = getFieldValue('enable_c_type_constraints');
                      return (
                        <Form.Item 
                          name="chang_type_ratio" 
                          label="长类比例 (%)"
                          tooltip="长类烟在每轮C类烟中的最大比例"
                        >
                          <InputNumber
                            min={0}
                            max={1}
                            step={0.05}
                            formatter={value => `${(value * 100).toFixed(0)}%`}
                            parser={value => value.replace('%', '') / 100}
                            style={{ width: '100%' }}
                            placeholder="20%"
                            disabled={!enabled}
                          />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item dependencies={['enable_c_type_constraints']} noStyle>
                    {({ getFieldValue }) => {
                      const enabled = getFieldValue('enable_c_type_constraints');
                      return (
                        <Form.Item 
                          name="chang_type_volume_limit" 
                          label="长类量限制 (箱)"
                          tooltip="每轮长类烟的最大限制"
                        >
                          <InputNumber
                            min={0}
                            step={100}
                            style={{ width: '100%' }}
                            placeholder="1000"
                            disabled={!enabled}
                          />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>
                </Col>
              </Row>
              
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item dependencies={['enable_c_type_constraints']} noStyle>
                    {({ getFieldValue }) => {
                      const enabled = getFieldValue('enable_c_type_constraints');
                      return (
                        <Form.Item 
                          name="xi_type_ratio" 
                          label="细类比例 (%)"
                          tooltip="细类烟在每轮C类烟中的最大比例"
                        >
                          <InputNumber
                            min={0}
                            max={1}
                            step={0.05}
                            formatter={value => `${(value * 100).toFixed(0)}%`}
                            parser={value => value.replace('%', '') / 100}
                            style={{ width: '100%' }}
                            placeholder="60%"
                            disabled={!enabled}
                          />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item dependencies={['enable_c_type_constraints']} noStyle>
                    {({ getFieldValue }) => {
                      const enabled = getFieldValue('enable_c_type_constraints');
                      return (
                        <Form.Item 
                          name="xi_type_volume_limit" 
                          label="细类量限制 (箱)"
                          tooltip="每轮细类烟的最大限制"
                        >
                          <InputNumber
                            min={0}
                            step={100}
                            style={{ width: '100%' }}
                            placeholder="3000"
                            disabled={!enabled}
                          />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>
                </Col>
              </Row>
              
              {/* 轮次约束配置 */}
              <Title level={5} style={{ marginTop: 16, marginBottom: 12 }}>
                <SettingOutlined style={{ marginRight: 8 }} />
                轮次约束配置
              </Title>
              <Alert
                message="轮次约束说明"
                description="投放总量总和不能变。"
                type="info"
                showIcon
                style={{ marginBottom: 16 }}
              />
              
              {uploadData && uploadData.rounds && uploadData.rounds.map((round, index) => (
                <Card 
                  key={round} 
                  size="small" 
                  title={`${round} 约束配置`}
                  style={{ marginBottom: 12 }}
                  bodyStyle={{ padding: '12px 16px' }}
                >
                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item 
                        name={`price_upper_limits_${round}`}
                        label="价格上限 (元)"
                        tooltip={`${round}轮次的单箱均价上限`}
                      >
                        <InputNumber
                          min={0}
                          step={10}
                          style={{ width: '100%' }}
                          placeholder={`默认: ${uploadData.round_constraints[round]?.upper_price_limit || '未设置'}`}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item 
                        name={`price_lower_limits_${round}`}
                        label="价格下限 (元)"
                        tooltip={`${round}轮次的单箱均价下限`}
                      >
                        <InputNumber
                          min={0}
                          step={10}
                          style={{ width: '100%' }}
                          placeholder={`默认: ${uploadData.round_constraints[round]?.lower_price_limit || '未设置'}`}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item 
                        name={`volume_limits_${round}`}
                        label="投放总量 (箱)"
                        tooltip={`${round}轮次的投放总量限制`}
                      >
                        <InputNumber
                          min={0}
                          step={100}
                          style={{ width: '100%' }}
                          placeholder={`默认: ${uploadData.round_constraints[round]?.total_quantity || '未设置'}`}
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                </Card>
              ))}
              
              {/* 其他配置参数 - 已移动到约束配置参数部分 */}
            </div>

            <Divider />

            <div className="config-section">
              <Title level={4}>目标函数权重配置</Title>
              <Alert
                message="优化目标说明"
                description="通过调整各项权重来控制优化目标的重要程度。权重越大，该目标在优化过程中的优先级越高。启用开关控制是否使用该目标。"
                type="info"
                showIcon
                style={{ marginBottom: 16 }}
              />
              
              {/* 目标函数配置 - 每行放置2个 */}
              <Row gutter={[16, 12]}>
                {/* 第一行：最大化总分配量 和 轮次间总量均衡 */}
                <Col span={12}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <Form.Item name="enable_maximize_allocation" valuePropName="checked" style={{ marginBottom: 0, marginRight: 8 }}>
                      <Switch size="small" checkedChildren="启用" unCheckedChildren="禁用" defaultChecked={true} />
                    </Form.Item>
                    <Text strong style={{ marginRight: 4 }}>最大化总分配量</Text>
                    <Tooltip title="控制尽量满足所有产品需求的重要程度，建议权重范围：1000-5000">
                      <QuestionCircleOutlined style={{ color: '#1890ff' }} />
                    </Tooltip>
                  </div>
                  <Form.Item dependencies={['enable_maximize_allocation']} style={{ marginTop: 4 }} noStyle>
                    {({ getFieldValue }) => {
                      const enabled = getFieldValue('enable_maximize_allocation');
                      return (
                        <Form.Item name="maximize_allocation_weight" style={{ marginBottom: 0 }}>
                          <InputNumber
                            min={0} max={5000} step={100} size="small"
                            style={{ width: '100%' }} placeholder="1000.0"
                            disabled={!enabled} addonBefore="权重"
                          />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>
                </Col>
                
                <Col span={12}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <Form.Item name="enable_round_balance" valuePropName="checked" style={{ marginBottom: 0, marginRight: 8 }}>
                      <Switch size="small" checkedChildren="启用" unCheckedChildren="禁用" defaultChecked={true} />
                    </Form.Item>
                    <Text strong style={{ marginRight: 4 }}>轮次间总量均衡</Text>
                    <Tooltip title="控制各轮次分配总量尽量均衡的重要程度，建议权重范围：500-2000">
                      <QuestionCircleOutlined style={{ color: '#1890ff' }} />
                    </Tooltip>
                  </div>
                  <Form.Item dependencies={['enable_round_balance']} style={{ marginTop: 4 }} noStyle>
                    {({ getFieldValue }) => {
                      const enabled = getFieldValue('enable_round_balance');
                      return (
                        <Form.Item name="round_balance_weight" style={{ marginBottom: 0 }}>
                          <InputNumber
                            min={0} max={2000} step={50} size="small"
                            style={{ width: '100%' }} placeholder="800.0"
                            disabled={!enabled} addonBefore="权重"
                          />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>
                </Col>

                {/* 第二行：轮次间方差最小化 和 品规级别均衡 */}
                <Col span={12}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <Form.Item name="enable_round_variance" valuePropName="checked" style={{ marginBottom: 0, marginRight: 8 }}>
                      <Switch size="small" checkedChildren="启用" unCheckedChildren="禁用" defaultChecked={true} />
                    </Form.Item>
                    <Text strong style={{ marginRight: 4 }}>轮次间方差最小化</Text>
                    <Tooltip title="控制减少各轮次分配量差异的重要程度，建议权重范围：200-1000">
                      <QuestionCircleOutlined style={{ color: '#1890ff' }} />
                    </Tooltip>
                  </div>
                  <Form.Item dependencies={['enable_round_variance']} style={{ marginTop: 4 }} noStyle>
                    {({ getFieldValue }) => {
                      const enabled = getFieldValue('enable_round_variance');
                      return (
                        <Form.Item name="round_variance_weight" style={{ marginBottom: 0 }}>
                          <InputNumber
                            min={0} max={1000} step={50} size="small"
                            style={{ width: '100%' }} placeholder="400.0"
                            disabled={!enabled} addonBefore="权重"
                          />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>
                </Col>
                
                <Col span={12}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <Form.Item name="enable_product_balance" valuePropName="checked" style={{ marginBottom: 0, marginRight: 8 }}>
                      <Switch size="small" checkedChildren="启用" unCheckedChildren="禁用" defaultChecked={true} />
                    </Form.Item>
                    <Text strong style={{ marginRight: 4 }}>品规级别均衡</Text>
                    <Tooltip title="控制品规在多轮分配、避免过度集中的重要程度，建议权重范围：50-500">
                      <QuestionCircleOutlined style={{ color: '#1890ff' }} />
                    </Tooltip>
                  </div>
                  <Form.Item dependencies={['enable_product_balance']} style={{ marginTop: 4 }} noStyle>
                    {({ getFieldValue }) => {
                      const enabled = getFieldValue('enable_product_balance');
                      return (
                        <Form.Item name="product_balance_weight" style={{ marginBottom: 0 }}>
                          <InputNumber
                            min={0} max={500} step={25} size="small"
                            style={{ width: '100%' }} placeholder="100.0"
                            disabled={!enabled} addonBefore="权重"
                          />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>
                </Col>

                {/* 第三行：轮次间平滑过渡 */}
                <Col span={12}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <Form.Item name="enable_smooth_transition" valuePropName="checked" style={{ marginBottom: 0, marginRight: 8 }}>
                      <Switch size="small" checkedChildren="启用" unCheckedChildren="禁用" defaultChecked={true} />
                    </Form.Item>
                    <Text strong style={{ marginRight: 4 }}>轮次间平滑过渡</Text>
                    <Tooltip title="控制相邻轮次分配量平滑变化的重要程度，建议权重范围：100-800">
                      <QuestionCircleOutlined style={{ color: '#1890ff' }} />
                    </Tooltip>
                  </div>
                  <Form.Item dependencies={['enable_smooth_transition']} style={{ marginTop: 4 }} noStyle>
                    {({ getFieldValue }) => {
                      const enabled = getFieldValue('enable_smooth_transition');
                      return (
                        <Form.Item name="smooth_transition_weight" style={{ marginBottom: 0 }}>
                          <InputNumber
                            min={0} max={800} step={50} size="small"
                            style={{ width: '100%' }} placeholder="300.0"
                            disabled={!enabled} addonBefore="权重"
                          />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>
                </Col>
              </Row>

              {/* 权重建议 */}
              <Alert
                message="权重配置建议"
                description="总分配量权重通常最高(1000+)，均衡类权重适中(100-800)，可根据实际需求调整各权重比例"
                type="info"
                showIcon
                style={{ marginTop: 16 }}
              />
            </div>
          </Form>
        </Card>
      )}

      {/* 计算按钮 */}
      {uploadData && (
        <Card title="3. 开始计算" style={{ marginBottom: 24 }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Button
              type="primary"
              size="large"
              icon={<PlayCircleOutlined />}
              loading={solving}
              onClick={handleSolve}
              style={{ width: 200 }}
            >
              {solving ? '计算中...' : '开始计算'}
            </Button>
            
            {solving && (
              <Alert
                message="正在计算中"
                description="请耐心等待，复杂的分配问题可能需要几分钟时间..."
                type="info"
                showIcon
              />
            )}
          </Space>
        </Card>
      )}

      {/* 计算结果 */}
      {solveResult && (
        <Card title="4. 计算结果" className="result-card">
          <Space direction="vertical" style={{ width: '100%' }}>
            <Alert
              message={
                <Space>
                  {solveResult.status === 'Optimal' ? (
                    <CheckCircleOutlined className="status-success" />
                  ) : (
                    <ExclamationCircleOutlined className="status-error" />
                  )}
                  <Text strong>
                    求解状态: {solveResult.status === 'Optimal' ? '最优解' : solveResult.status}
                  </Text>
                </Space>
              }
              type={solveResult.status === 'Optimal' ? 'success' : 'error'}
              showIcon={false}
            />
            
            {solveResult.status === 'Optimal' && (
              <Descriptions bordered column={2}>
                <Descriptions.Item label="目标函数值">
                  {solveResult.objective_value?.toFixed(2) || 'N/A'}
                </Descriptions.Item>
                <Descriptions.Item label="求解时间">
                  {solveResult.solve_time?.toFixed(2) || 'N/A'} 秒
                </Descriptions.Item>
                <Descriptions.Item label="总分配量">
                  {solveResult.total_allocated?.toFixed(2) || 'N/A'} 箱
                </Descriptions.Item>
                <Descriptions.Item label="分配率">
                  {solveResult.summary?.allocation_rate ? 
                    `${(solveResult.summary.allocation_rate * 100).toFixed(1)}%` : 'N/A'}
                </Descriptions.Item>
              </Descriptions>
            )}
            
            <Alert
              message="计算完成"
              description="请前往「分配明细查看 & 导出」页面查看详细结果，或前往「计算结果页面」查看汇总信息。"
              type="success"
              showIcon
              action={
                <Space>
                  <Button size="small" onClick={() => navigate('/allocation')}>
                    查看明细
                  </Button>
                  <Button size="small" onClick={() => navigate('/result')}>
                    查看约束
                  </Button>
                </Space>
              }
            />
          </Space>
        </Card>
      )}
    </div>
  );
};

export default HomePage;