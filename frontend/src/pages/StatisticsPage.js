import React, { useState, useEffect } from 'react';
import { Card, Table, Row, Col, Typography, Spin, Alert, Button, message } from 'antd';
import { BarChartOutlined, DownloadOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { getResult } from '../services/api';

const { Title, Text } = Typography;

/**
 * 结果统计页面组件
 * 展示分配结果的统计信息，包括各轮次和总计的统计数据
 */
const StatisticsPage = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [resultData, setResultData] = useState(null);
  const [statisticsData, setStatisticsData] = useState([]);
  const navigate = useNavigate();

  // 导出Excel统计表
  const handleExportExcel = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/export-statistics?format=xlsx');
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        
        // 生成带时间戳的文件名
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        a.download = `分配结果统计表_${timestamp}.xlsx`;
        
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        message.success('统计表导出成功！');
      } else {
        const errorData = await response.json();
        message.error(errorData.detail || '统计表导出失败');
      }
    } catch (error) {
      console.error('导出统计表失败:', error);
      message.error('统计表导出失败');
    }
  };

  /**
   * 计算统计数据
   * @param {Object} data - 分配结果数据
   */
  const calculateStatistics = React.useCallback((data) => {
    const { allocation_details } = data;
    
    // 获取轮次列表
    const rounds = [];
    if (allocation_details.length > 0) {
      Object.keys(allocation_details[0]).forEach(key => {
        if (key.startsWith('allocation_')) {
          const roundName = key.replace('allocation_', '');
          rounds.push(roundName);
        }
      });
    }

    // 计算统计数据
    const stats = [];

    // 1. 单箱（单箱均价）
    const unitPriceRow = { metric: '单箱', total: 0 };
    rounds.forEach(round => {
      const roundKey = `allocation_${round}`;
      let totalAllocation = 0;
      let totalValue = 0;
      
      allocation_details.forEach(item => {
        const allocation = item[roundKey] || 0;
        const unitPrice = item.unit_price || 0;
        totalAllocation += allocation;
        totalValue += allocation * unitPrice;
      });
      
      unitPriceRow[round] = totalAllocation > 0 ? (totalValue / totalAllocation).toFixed(1) : '0.0';
    });
    
    // 总计单箱价格
    let totalAllocationAll = 0;
    let totalValueAll = 0;
    allocation_details.forEach(item => {
      const allocation = item.total_allocation || 0;
      const unitPrice = item.unit_price || 0;
      totalAllocationAll += allocation;
      totalValueAll += allocation * unitPrice;
    });
    unitPriceRow.total = totalAllocationAll > 0 ? (totalValueAll / totalAllocationAll).toFixed(1) : '0.0';
    stats.push(unitPriceRow);

    // 2. 总分配量
    const totalAllocationRow = { metric: '总量', total: 0 };
    rounds.forEach(round => {
      const roundKey = `allocation_${round}`;
      const roundTotal = allocation_details.reduce((sum, item) => sum + (item[roundKey] || 0), 0);
      totalAllocationRow[round] = roundTotal.toFixed(2);
    });
    totalAllocationRow.total = allocation_details.reduce((sum, item) => sum + (item.total_allocation || 0), 0).toFixed(2);
    stats.push(totalAllocationRow);

    // 3. 总数（投放大于0的品规数量）
    const totalCountRow = { metric: '总数', total: 0 };
    rounds.forEach(round => {
      const roundKey = `allocation_${round}`;
      const count = allocation_details.filter(item => (item[roundKey] || 0) > 0).length;
      totalCountRow[round] = count;
    });
    totalCountRow.total = allocation_details.filter(item => (item.total_allocation || 0) > 0).length;
    stats.push(totalCountRow);

    // 4. C类量（C列有值的分配量）
    const cCategoryRow = { metric: 'C类量', total: 0 };
    rounds.forEach(round => {
      const roundKey = `allocation_${round}`;
      const cTotal = allocation_details
        .filter(item => item.c_type && item.c_type.trim() !== '')
        .reduce((sum, item) => sum + (item[roundKey] || 0), 0);
      cCategoryRow[round] = cTotal.toFixed(2);
    });
    cCategoryRow.total = allocation_details
      .filter(item => item.c_type && item.c_type.trim() !== '')
      .reduce((sum, item) => sum + (item.total_allocation || 0), 0).toFixed(2);
    stats.push(cCategoryRow);

    // 5. C类占比
    const cRatioRow = { metric: '占比', total: 0 };
    rounds.forEach(round => {
      const roundTotal = parseFloat(totalAllocationRow[round]);
      const cTotal = parseFloat(cCategoryRow[round]);
      cRatioRow[round] = roundTotal > 0 ? ((cTotal / roundTotal) * 100).toFixed(1) + '%' : '0.0%';
    });
    const totalTotal = parseFloat(totalAllocationRow.total);
    const cTotalAll = parseFloat(cCategoryRow.total);
    cRatioRow.total = totalTotal > 0 ? ((cTotalAll / totalTotal) * 100).toFixed(1) + '%' : '0.0%';
    stats.push(cRatioRow);

    // 6. C类数（C类投放大于0的品规数量）
    const cCountRow = { metric: 'C类数', total: 0 };
    rounds.forEach(round => {
      const roundKey = `allocation_${round}`;
      const count = allocation_details.filter(item => 
        item.c_type && item.c_type.trim() !== '' && (item[roundKey] || 0) > 0
      ).length;
      cCountRow[round] = count;
    });
    cCountRow.total = allocation_details.filter(item => 
      item.c_type && item.c_type.trim() !== '' && (item.total_allocation || 0) > 0
    ).length;
    stats.push(cCountRow);

    // 7-9. 方块、长片、细支类似处理
    const categories = [
      { name: '方块', key: 'square', match: '方' },
      { name: '长片', key: 'long', match: '长' },
      { name: '细支', key: 'thin', match: '细' }
    ];

    categories.forEach(category => {
      // 量
      const categoryAmountRow = { metric: `${category.name}量`, total: 0 };
      rounds.forEach(round => {
        const roundKey = `allocation_${round}`;
        const categoryTotal = allocation_details
          .filter(item => item.c_category && item.c_category.includes(category.match))
          .reduce((sum, item) => sum + (item[roundKey] || 0), 0);
        categoryAmountRow[round] = categoryTotal.toFixed(2);
      });
      categoryAmountRow.total = allocation_details
        .filter(item => item.c_category && item.c_category.includes(category.match))
        .reduce((sum, item) => sum + (item.total_allocation || 0), 0).toFixed(2);
      stats.push(categoryAmountRow);

      // 数量
      const categoryCountRow = { metric: `${category.name}数`, total: 0 };
      rounds.forEach(round => {
        const roundKey = `allocation_${round}`;
        const count = allocation_details.filter(item => 
          item.c_category && item.c_category.includes(category.match) && (item[roundKey] || 0) > 0
        ).length;
        categoryCountRow[round] = count;
      });
      categoryCountRow.total = allocation_details.filter(item => 
        item.c_category && item.c_category.includes(category.match) && (item.total_allocation || 0) > 0
      ).length;
      stats.push(categoryCountRow);

      // 占比（占C类的占比）
      const categoryRatioRow = { metric: `${category.name}占比`, total: 0 };
      rounds.forEach(round => {
        const cTotal = parseFloat(cCategoryRow[round]);
        const categoryTotal = parseFloat(categoryAmountRow[round]);
        categoryRatioRow[round] = cTotal > 0 ? ((categoryTotal / cTotal) * 100).toFixed(1) + '%' : '0.0%';
      });
      const cTotalAll = parseFloat(cCategoryRow.total);
      const categoryTotalAll = parseFloat(categoryAmountRow.total);
      categoryRatioRow.total = cTotalAll > 0 ? ((categoryTotalAll / cTotalAll) * 100).toFixed(1) + '%' : '0.0%';
      stats.push(categoryRatioRow);
    });

    setStatisticsData(stats);
  }, []);

  /**
   * 获取分配结果数据
   */
  const fetchData = React.useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await getResult();
      // 修正：axios返回的数据在response.data中
      const responseData = response.data;
      if (responseData.success) {
        setResultData(responseData.data);
        calculateStatistics(responseData.data);
      } else {
        setError('获取数据失败：' + (responseData.message || '未知错误'));
      }
    } catch (error) {
      console.error('获取数据失败:', error);
      if (error.response && error.response.status === 400) {
        // 400错误通常表示暂无计算结果
        setError(null);
        setResultData(null);
      } else {
        setError('网络错误或服务器异常，请检查后端服务是否正常运行');
      }
    } finally {
      setLoading(false);
    }
  }, [calculateStatistics]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  /**
   * 获取表格列定义
   */
  const getColumns = () => {
    const columns = [
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>指标</span>,
        dataIndex: 'metric',
        key: 'metric',
        width: 120,
        fixed: 'left',
        render: (value) => <span style={{ fontSize: '15px', fontWeight: 'bold' }}>{value}</span>
      }
    ];

    // 添加轮次列
    if (resultData && resultData.allocation_details.length > 0) {
      Object.keys(resultData.allocation_details[0]).forEach(key => {
        if (key.startsWith('allocation_') && key !== 'allocation_rate') {
          const roundName = key.replace('allocation_', '');
          columns.push({
            title: <span style={{ fontSize: '16px', fontWeight: 'bold', color: '#1890ff' }}>{roundName}</span>,
            dataIndex: roundName,
            key: roundName,
            width: 120,
            render: (value) => <span style={{ fontSize: '15px' }}>{value}</span>
          });
        }
      });
    }

    // 添加总计列
    columns.push({
      title: <span style={{ fontSize: '16px', fontWeight: 'bold', color: '#52c41a' }}>总计</span>,
      dataIndex: 'total',
      key: 'total',
      width: 120,
      render: (value) => <span style={{ fontSize: '15px', fontWeight: 'bold', color: '#52c41a' }}>{value}</span>
    });

    return columns;
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>
          <Text style={{ fontSize: '16px' }}>正在检查计算结果...</Text>
        </div>
        <div style={{ marginTop: 8 }}>
          <Text type="secondary">如果长时间无响应，请确保已完成数据计算</Text>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '50px', textAlign: 'center' }}>
        <Alert
          message="获取统计数据失败"
          description={error}
          type="error"
          showIcon
          action={
            <div>
              <Button type="primary" onClick={fetchData} style={{ marginRight: 8 }}>
                重试
              </Button>
              <Button onClick={() => navigate('/')}>
                返回首页
              </Button>
            </div>
          }
          style={{ 
            maxWidth: '600px', 
            margin: '0 auto',
            textAlign: 'left'
          }}
        />
      </div>
    );
  }

  if (!resultData) {
    return (
      <div style={{ padding: '50px', textAlign: 'center' }}>
        <Alert
          message="暂无统计数据"
          description={
            <div style={{ textAlign: 'left', marginTop: '16px' }}>
              <p>要查看结果统计分析，请按以下步骤操作：</p>
              <ol style={{ paddingLeft: '20px' }}>
                <li>在首页上传Excel数据文件</li>
                <li>配置计算参数（约束条件和目标函数权重）</li>
                <li><strong>点击"开始计算"按钮执行分配计算</strong></li>
                <li>计算完成后返回此页面查看统计结果</li>
              </ol>
              <p style={{ marginTop: '16px', color: '#1890ff' }}>
                💡 提示：如果您已经上传了文件，请确保已经点击了"开始计算"按钮。
              </p>
            </div>
          }
          type="info"
          showIcon
          action={
            <Button type="primary" size="large" onClick={() => navigate('/')}>
              前往首页开始计算
            </Button>
          }
          style={{ 
            maxWidth: '600px', 
            margin: '0 auto',
            textAlign: 'left'
          }}
        />
      </div>
    );
  }

  return (
    <div>
      <Title level={2} style={{ fontSize: '32px', marginBottom: '32px' }}>
        <BarChartOutlined style={{ marginRight: 12, fontSize: '36px' }} />
        结果统计分析
      </Title>

      {/* 统计表格 */}
      <Card 
        title={
          <span style={{ 
            fontSize: '24px', 
            fontWeight: 'bold',
            background: 'linear-gradient(45deg, #1890ff, #52c41a)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text'
          }}>
            📊 分配结果统计表
          </span>
        }
        extra={
          <Button 
            type="primary" 
            icon={<DownloadOutlined />}
            onClick={handleExportExcel}
            size="large"
          >
            导出Excel
          </Button>
        }
        style={{ 
          marginBottom: 32,
          background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
          borderRadius: '12px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.12)'
        }}
      >
        <Table
          columns={getColumns()}
          dataSource={statisticsData.map((item, index) => ({ ...item, key: index }))}
          pagination={false}
          scroll={{ x: 1200 }}
          size="large"
          style={{ fontSize: '16px' }}
          bordered
        />
      </Card>

      {/* 说明信息 */}
      <Card title="统计说明" style={{ marginTop: 24 }}>
        <Row gutter={24}>
          <Col span={12}>
            <div style={{ marginBottom: 16 }}>
              <Text strong>指标说明：</Text>
            </div>
            <ul style={{ paddingLeft: 20 }}>
              <li><Text>单箱：各轮次和总计的单箱均价</Text></li>
              <li><Text>总量：总分配量（保留两位小数）</Text></li>
              <li><Text>总数：投放大于0的品规数量</Text></li>
              <li><Text>C类量：C列有值的分配量</Text></li>
              <li><Text>占比：C类量/总量的百分比</Text></li>
              <li><Text>C类数：C类投放大于0的品规数量</Text></li>
            </ul>
          </Col>
          <Col span={12}>
            <div style={{ marginBottom: 16 }}>
              <Text strong>分类统计：</Text>
            </div>
            <ul style={{ paddingLeft: 20 }}>
              <li><Text>方块量/数：方块类型的分配量和数量</Text></li>
              <li><Text>方块占比：方块量占C类量的百分比</Text></li>
              <li><Text>长片量/数：长片类型的分配量和数量</Text></li>
              <li><Text>长片占比：长片量占C类量的百分比</Text></li>
              <li><Text>细支量/数：细支类型的分配量和数量</Text></li>
              <li><Text>细支占比：细支量占C类量的百分比</Text></li>
            </ul>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default StatisticsPage;