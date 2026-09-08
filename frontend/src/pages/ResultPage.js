import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  Spin,
  Alert,
  Typography,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Button,
  Space,
  Collapse,
  List,
  message
} from 'antd';
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined,
  WarningOutlined
} from '@ant-design/icons';
import { getConstraints } from '../services/api';

const { Title, Text } = Typography;
const { Panel } = Collapse;

/**
 * 约束验证页面组件
 * 显示详细的约束验证结果和违反情况
 */
const ConstraintValidationPage = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [constraintData, setConstraintData] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  // 组件挂载时获取数据
  useEffect(() => {
    fetchData();
  }, []);

  /**
   * 获取约束验证数据
   */
  const fetchData = async () => {
    try {
      const response = await getConstraints();

      if (response.data.success) {
        setConstraintData(response.data.data);
      } else {
        message.error(response.data.message || '获取约束验证结果失败');
      }
    } catch (error) {
      console.error('获取约束验证数据失败:', error);
      message.error('获取约束验证数据失败，请先完成计算');
    } finally {
      setLoading(false);
    }
  };

  /**
   * 刷新数据
   */
  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchData();
    setRefreshing(false);
    message.success('数据已刷新');
  };

  /**
   * 获取约束类型的中文名称
   */
  const getConstraintTypeName = (type) => {
    const typeMap = {
      'demand': '需求满足约束',
      'price': '单箱均价约束',
      'volume': '投放总量约束',
      'fixed_allocation': '固定分配约束',
      'first_round_supply': '第一轮货源上限约束',
      'demand_based_priority': '按需品规优先约束',
      'price_based': '按价品规比例约束',
      'c_type': 'C类烟约束'
    };
    return typeMap[type] || type;
  };

  /**
   * 获取整体验证状态的显示文本和样式
   */
  const getOverallStatus = (constraintData) => {
    const summary = constraintData.summary || {};
    const totalViolations = summary.total_violations || 0;
    
    if (totalViolations === 0) {
      return {
        text: '全部通过',
        color: '#3f8600',
        icon: <CheckCircleOutlined />
      };
    } else {
      return {
        text: '部分通过',
        color: '#fa8c16',
        icon: <ExclamationCircleOutlined />
      };
    }
  };

  /**
   * 获取约束状态的颜色和图标
   */
  const getConstraintStatus = (isValid) => {
    return {
      color: isValid ? 'green' : 'red',
      icon: isValid ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />,
      text: isValid ? '通过' : '违反'
    };
  };

  /**
   * 渲染违反详情
   */
  const renderViolationDetails = (constraint) => {
    if (!constraint.violations || constraint.violations.length === 0) {
      return <Text type="success">无违反情况</Text>;
    }

    return (
      <List
        size="small"
        dataSource={constraint.violations}
        renderItem={(violation, index) => (
          <List.Item>
            <Alert
              message={`违反 ${index + 1}`}
              description={
                <div style={{ fontSize: '14px' }}>
                  {Object.entries(violation).map(([key, value]) => (
                    <div key={key} style={{ marginBottom: '4px' }}>
                      <Text strong>{key}:</Text> {typeof value === 'number' ? value.toFixed(4) : String(value)}
                    </div>
                  ))}
                </div>
              }
              type="error"
              size="small"
              showIcon
              style={{ marginBottom: '8px' }}
            />
          </List.Item>
        )}
      />
    );
  };

  /**
   * 约束验证表格的列定义
   */
  const constraintColumns = [
    {
      title: '约束类型',
      dataIndex: 'type',
      key: 'type',
      width: 180,
      render: (type) => (
        <Text strong style={{ fontSize: '16px' }}>
          {getConstraintTypeName(type)}
        </Text>
      )
    },
    {
      title: '验证状态',
      dataIndex: 'is_valid',
      key: 'is_valid',
      width: 120,
      render: (isValid) => {
        const status = getConstraintStatus(isValid);
        return (
          <Tag 
            color={status.color} 
            icon={status.icon}
            style={{ fontSize: '14px', padding: '4px 8px' }}
          >
            {status.text}
          </Tag>
        );
      }
    },
    {
      title: '违反数量',
      dataIndex: 'violations_count',
      key: 'violations_count',
      width: 100,
      render: (count) => (
        <Statistic
          value={count || 0}
          valueStyle={{ 
            fontSize: '16px',
            color: count > 0 ? '#ff4d4f' : '#52c41a'
          }}
        />
      )
    }
  ];

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>
          <Text style={{ fontSize: '16px' }}>正在加载约束验证结果...</Text>
        </div>
      </div>
    );
  }

  if (!constraintData) {
    return (
      <Alert
        message="暂无约束验证结果"
        description="请先在首页完成数据上传和计算配置，然后开始计算。"
        type="warning"
        showIcon
        action={
          <Button type="primary" onClick={() => navigate('/')}>
            返回首页
          </Button>
        }
      />
    );
  }

  // 准备表格数据
  const tableData = Object.entries(constraintData.constraint_results || {}).map(([type, result]) => ({
    key: type,
    type,
    is_valid: result.is_valid,
    violations_count: result.violations ? result.violations.length : 0,
    constraint_data: result
  }));

  const summary = constraintData.summary || {};

  return (
    <div>
      <Title level={2} style={{ fontSize: '28px', marginBottom: '24px' }}>
        <WarningOutlined style={{ marginRight: 8 }} />
        约束验证结果
        <Button 
          type="text" 
          icon={<ReloadOutlined />} 
          loading={refreshing}
          onClick={handleRefresh}
          style={{ float: 'right', fontSize: '16px' }}
        >
          刷新数据
        </Button>
      </Title>

      {/* 总体验证状态 */}
      <Card 
        title={<span style={{ fontSize: '20px' }}>总体验证状态</span>} 
        style={{ marginBottom: 24 }}
      >
        <Row gutter={24}>
          <Col span={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic
                title={<span style={{ fontSize: '16px' }}>整体状态</span>}
                value={getOverallStatus(constraintData).text}
                valueStyle={{ 
                  fontSize: '24px',
                  color: getOverallStatus(constraintData).color,
                  fontWeight: 'bold'
                }}
                prefix={getOverallStatus(constraintData).icon}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic
                title={<span style={{ fontSize: '16px' }}>总违反数</span>}
                value={summary.total_violations || 0}
                valueStyle={{ 
                  fontSize: '24px',
                  color: summary.total_violations > 0 ? '#cf1322' : '#3f8600'
                }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic
                title={<span style={{ fontSize: '16px' }}>通过约束</span>}
                value={summary.passed_constraints ? summary.passed_constraints.length : 0}
                valueStyle={{ fontSize: '24px', color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic
                title={<span style={{ fontSize: '16px' }}>违反约束</span>}
                value={summary.violated_constraints ? summary.violated_constraints.length : 0}
                valueStyle={{ fontSize: '24px', color: '#cf1322' }}
              />
            </Card>
          </Col>
        </Row>

        {/* 约束摘要信息 */}
        <Row gutter={16} style={{ marginTop: 24 }}>
          <Col span={12}>
            <Alert
              message={<span style={{ fontSize: '16px' }}>启用的约束</span>}
              description={
                <div style={{ fontSize: '14px' }}>
                  {summary.enabled_constraints && summary.enabled_constraints.length > 0 ? (
                    summary.enabled_constraints.map(constraint => (
                      <Tag key={constraint} color="blue" style={{ margin: '2px', fontSize: '12px' }}>
                        {getConstraintTypeName(constraint)}
                      </Tag>
                    ))
                  ) : (
                    <Text>无</Text>
                  )}
                </div>
              }
              type="info"
              showIcon
            />
          </Col>
          <Col span={12}>
            <Alert
              message={<span style={{ fontSize: '16px' }}>跳过的约束</span>}
              description={
                <div style={{ fontSize: '14px' }}>
                  {summary.skipped_constraints && summary.skipped_constraints.length > 0 ? (
                    summary.skipped_constraints.map(constraint => (
                      <Tag key={constraint} color="default" style={{ margin: '2px', fontSize: '12px' }}>
                        {getConstraintTypeName(constraint)}
                      </Tag>
                    ))
                  ) : (
                    <Text>无</Text>
                  )}
                </div>
              }
              type="warning"
              showIcon
            />
          </Col>
        </Row>
      </Card>

      {/* 约束验证详情表格 */}
      <Card 
        title={<span style={{ fontSize: '20px' }}>约束验证详情</span>} 
        style={{ marginBottom: 24 }}
      >
        <Table
          columns={constraintColumns}
          dataSource={tableData}
          pagination={false}
          size="middle"
          style={{ fontSize: '14px' }}
        />
      </Card>

      {/* 违反详情展开面板 */}
      {tableData.some(item => item.violations_count > 0) && (
        <Card title={<span style={{ fontSize: '20px' }}>违反详情</span>}>
          <Collapse>
            {tableData
              .filter(item => item.violations_count > 0)
              .map(item => (
                <Panel
                  header={
                    <Space>
                      <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
                      <Text strong style={{ fontSize: '16px' }}>
                        {getConstraintTypeName(item.type)} 
                      </Text>
                      <Tag color="red" style={{ fontSize: '12px' }}>
                        {item.violations_count} 个违反
                      </Tag>
                    </Space>
                  }
                  key={item.type}
                >
                  {renderViolationDetails(item.constraint_data)}
                </Panel>
              ))}
          </Collapse>
        </Card>
      )}
    </div>
  );
};

export default ConstraintValidationPage;