import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  Table,
  Button,
  Space,
  message,
  Spin,
  Alert,
  Typography,
  Tag,
  Row,
  Col,
  Statistic,
  Input
} from 'antd';
import {
  DownloadOutlined,
  EyeOutlined,
  SearchOutlined
} from '@ant-design/icons';
import { getResult, exportResult } from '../services/api';

const { Title, Text } = Typography;
const { Search } = Input;

const AllocationPage = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [resultData, setResultData] = useState(null);
  const [allocationMatrix, setAllocationMatrix] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [searchText, setSearchText] = useState('');
  const [exporting, setExporting] = useState(false);
  const [rounds, setRounds] = useState([]);

  // 获取结果数据
  useEffect(() => {
    fetchResult();
  }, []);

  const fetchResult = async () => {
    try {
      const response = await getResult();
      if (response.data.success) {
        const data = response.data.data;
        setResultData(data);
        
        // 处理分配矩阵数据 - 完全按照output4.xlsx的结构
        const matrixData = data.allocation_details.map((item, index) => {
          const matrixRow = {
            key: index,
            product_code: item.product_code,
            wholesale_price: item.wholesale_price,
            product_name: item.product_name,
            category: item.category,
            demand: item.demand,
            available_supply: item.available_supply,
            attribute: item.attribute,
            c_category: item.c_category,
            brand: item.brand,
            stick_ratio: item.stick_ratio,
            c_type: item.c_type,
            demand_based: item.demand_based,
            price_based: item.price_based,
            unit_price: item.unit_price,
            total_allocation: item.total_allocation,
            allocation_rate: item.allocation_rate
          };
          
          // 添加各轮次分配量
          Object.keys(item).forEach(key => {
            if (key.startsWith('allocation_')) {
              const roundName = key.replace('allocation_', '');
              matrixRow[roundName] = item[key];
            }
          });
          
          return matrixRow;
        });
        
        setAllocationMatrix(matrixData);
        setFilteredData(matrixData);
        
        // 提取轮次信息
        const roundNames = data.round_summary.map(round => round.round_name);
        setRounds(roundNames);
      } else {
        message.error(response.data.message || '获取结果失败');
      }
    } catch (error) {
      console.error('获取结果失败:', error);
      message.error('获取结果失败，请先完成计算');
    } finally {
      setLoading(false);
    }
  };

  // 搜索过滤
  useEffect(() => {
    if (!allocationMatrix.length) return;
    
    let filtered = allocationMatrix;
    
    // 按产品名称或代码搜索
    if (searchText) {
      filtered = filtered.filter(item =>
        (String(item.product_name || '')).toLowerCase().includes(searchText.toLowerCase()) ||
        (String(item.product_code || '')).toLowerCase().includes(searchText.toLowerCase())
      );
    }
    
    setFilteredData(filtered);
  }, [searchText, allocationMatrix]);

  // 导出结果
  const handleExport = async (format) => {
    setExporting(true);
    try {
      const response = await exportResult(format);
      
      // 创建下载链接
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `allocation_result.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      message.success(`${format.toUpperCase()} 文件导出成功`);
    } catch (error) {
      console.error('导出失败:', error);
      message.error('导出失败');
    } finally {
      setExporting(false);
    }
  };

  // 表格列定义 - 调整列顺序，轮次数据移到可用货源后面
  const getColumns = () => {
    const columns = [
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>代码</span>,
        dataIndex: 'product_code',
        key: 'product_code',
        width: 120,
        fixed: 'left',
        render: (value) => <span style={{ fontSize: '15px' }}>{value}</span>
      },
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>批发价</span>,
        dataIndex: 'wholesale_price',
        key: 'wholesale_price',
        width: 100,
        render: (value) => <span style={{ fontSize: '15px' }}>{(value || 0).toFixed(1)}</span>
      },
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>卷烟名称</span>,
        dataIndex: 'product_name',
        key: 'product_name',
        width: 200,
        fixed: 'left',
        render: (value) => <span style={{ fontSize: '15px' }}>{value}</span>
      },
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>类</span>,
        dataIndex: 'category',
        key: 'category',
        width: 60,
        render: (value) => <span style={{ fontSize: '15px' }}>{value}</span>
      },
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>需求</span>,
        dataIndex: 'demand',
        key: 'demand',
        width: 100,
        render: (value) => <span style={{ fontSize: '15px', fontWeight: 'bold' }}>{(value || 0).toFixed(3)}</span>
      },
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>可用货源</span>,
        dataIndex: 'available_supply',
        key: 'available_supply',
        width: 120,
        render: (value) => <span style={{ fontSize: '15px' }}>{(value || 0).toFixed(3)}</span>
      }
    ];

    // 添加轮次列（移到可用货源后面）
    const roundColumns = rounds.map(roundName => ({
      title: <span style={{ fontSize: '16px', fontWeight: 'bold', color: '#1890ff' }}>{roundName}</span>,
      dataIndex: roundName,
      key: roundName,
      width: 100,
      render: (value) => (
        <span style={{ 
          fontSize: '15px', 
          fontWeight: value > 0 ? 'bold' : 'normal',
          color: value > 0 ? '#1890ff' : '#999'
        }}>
          {(value || 0).toFixed(3)}
        </span>
      )
    }));

    // 添加总分配量和分配率列（紧跟轮次后面）
    const allocationSummaryColumns = [
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold', color: '#52c41a' }}>总分配量</span>,
        dataIndex: 'total_allocation',
        key: 'total_allocation',
        width: 120,
        render: (value) => <span style={{ fontSize: '15px', fontWeight: 'bold', color: '#52c41a' }}>{(value || 0).toFixed(3)}</span>
      },
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>分配率</span>,
        dataIndex: 'allocation_rate',
        key: 'allocation_rate',
        width: 100,
        render: (value) => {
          const rate = value || 0;
          return (
            <Tag 
              color={rate >= 0.9 ? 'green' : rate >= 0.7 ? 'orange' : 'red'}
              style={{ fontSize: '14px', padding: '4px 8px', fontWeight: 'bold' }}
            >
              {rate.toFixed(1)}
            </Tag>
          );
        }
      }
    ];

    // 添加其他属性列
    const attributeColumns = [
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>属</span>,
        dataIndex: 'attribute',
        key: 'attribute',
        width: 80,
        render: (value) => <span style={{ fontSize: '15px' }}>{value || ''}</span>
      },
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>C类</span>,
        dataIndex: 'c_category',
        key: 'c_category',
        width: 80,
        render: (value) => <span style={{ fontSize: '15px' }}>{value || ''}</span>
      },
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>品牌</span>,
        dataIndex: 'brand',
        key: 'brand',
        width: 120,
        render: (value) => <span style={{ fontSize: '15px' }}>{value || ''}</span>
      },
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>条支比</span>,
        dataIndex: 'stick_ratio',
        key: 'stick_ratio',
        width: 100,
        render: (value) => <span style={{ fontSize: '15px' }}>{value || ''}</span>
      },
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>C</span>,
        dataIndex: 'c_type',
        key: 'c_type',
        width: 60,
        render: (value) => <span style={{ fontSize: '15px' }}>{value || ''}</span>
      },
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>按需</span>,
        dataIndex: 'demand_based',
        key: 'demand_based',
        width: 80,
        render: (value) => <span style={{ fontSize: '15px' }}>{value || ''}</span>
      },
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>按价</span>,
        dataIndex: 'price_based',
        key: 'price_based',
        width: 80,
        render: (value) => <span style={{ fontSize: '15px' }}>{value || ''}</span>
      }
    ];

    // 添加单箱价格列
    const unitPriceColumns = [
      {
        title: <span style={{ fontSize: '16px', fontWeight: 'bold' }}>单箱价格</span>,
        dataIndex: 'unit_price',
        key: 'unit_price',
        width: 120,
        render: (value) => <span style={{ fontSize: '15px' }}>{(value || 0).toFixed(1)}</span>
      }
    ];

    return [...columns, ...roundColumns, ...allocationSummaryColumns, ...attributeColumns, ...unitPriceColumns];
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>
          <Text>正在加载分配结果...</Text>
        </div>
      </div>
    );
  }

  if (!resultData) {
    return (
      <Alert
        message="暂无分配结果"
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

  return (
    <div>
      <Title level={2} style={{ fontSize: '32px', marginBottom: '32px' }}>
        <EyeOutlined style={{ marginRight: 12, fontSize: '36px' }} />
        分配明细查看 & 导出
      </Title>

      {/* 分配汇总 - 美化版本 */}
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
            📊 分配汇总
          </span>
        } 
        style={{ 
          marginBottom: 32,
          background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
          borderRadius: '12px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.12)'
        }}
      >
        <Row gutter={24}>
          <Col span={6}>
            <div style={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              borderRadius: '12px',
              padding: '24px',
              textAlign: 'center',
              color: 'white',
              boxShadow: '0 4px 12px rgba(102, 126, 234, 0.4)'
            }}>
              <div style={{ fontSize: '16px', marginBottom: '8px', opacity: 0.9 }}>总数</div>
              <div style={{ fontSize: '32px', fontWeight: 'bold' }}>
                {allocationMatrix.filter(item => (item.total_allocation || 0) > 0).length}
              </div>
              <div style={{ fontSize: '14px', opacity: 0.8 }}>有分配的品规数量</div>
            </div>
          </Col>
          <Col span={6}>
            <div style={{
              background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
              borderRadius: '12px',
              padding: '24px',
              textAlign: 'center',
              color: 'white',
              boxShadow: '0 4px 12px rgba(240, 147, 251, 0.4)'
            }}>
              <div style={{ fontSize: '16px', marginBottom: '8px', opacity: 0.9 }}>总分配量</div>
              <div style={{ fontSize: '32px', fontWeight: 'bold' }}>
                {allocationMatrix.reduce((sum, item) => sum + (item.total_allocation || 0), 0).toFixed(3)}
              </div>
              <div style={{ fontSize: '14px', opacity: 0.8 }}>箱</div>
            </div>
          </Col>
          <Col span={6}>
            <div style={{
              background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
              borderRadius: '12px',
              padding: '24px',
              textAlign: 'center',
              color: 'white',
              boxShadow: '0 4px 12px rgba(79, 172, 254, 0.4)'
            }}>
              <div style={{ fontSize: '16px', marginBottom: '8px', opacity: 0.9 }}>整体分配率</div>
              <div style={{ fontSize: '32px', fontWeight: 'bold' }}>
                {
                  (() => {
                    const totalDemand = allocationMatrix.reduce((sum, item) => sum + (item.demand || 0), 0);
                    const totalAllocation = allocationMatrix.reduce((sum, item) => sum + (item.total_allocation || 0), 0);
                    return totalDemand > 0 ? (totalAllocation / totalDemand * 100).toFixed(1) : '0.0';
                  })()
                }%
              </div>
              <div style={{ fontSize: '14px', opacity: 0.8 }}>分配完成度</div>
            </div>
          </Col>
          <Col span={6}>
            <div style={{
              background: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
              borderRadius: '12px',
              padding: '24px',
              textAlign: 'center',
              color: 'white',
              boxShadow: '0 4px 12px rgba(250, 112, 154, 0.4)'
            }}>
              <div style={{ fontSize: '16px', marginBottom: '8px', opacity: 0.9 }}>单箱</div>
              <div style={{ fontSize: '32px', fontWeight: 'bold' }}>
                {
                  (() => {
                    const totalAllocation = allocationMatrix.reduce((sum, item) => sum + (item.total_allocation || 0), 0);
                    const totalValue = allocationMatrix.reduce((sum, item) => {
                      const allocation = item.total_allocation || 0;
                      const unitPrice = item.unit_price || 0;
                      return sum + (allocation * unitPrice);
                    }, 0);
                    return totalAllocation > 0 ? (totalValue / totalAllocation).toFixed(1) : '0.0';
                  })()
                }
              </div>
              <div style={{ fontSize: '14px', opacity: 0.8 }}>整体单箱价格</div>
            </div>
          </Col>
        </Row>
      </Card>

      {/* 分配明细表格 */}
      <Card title={<span style={{ fontSize: '20px' }}>分配明细矩阵</span>} style={{ marginBottom: 32 }}>
        <Row gutter={24} style={{ marginBottom: 24 }}>
          <Col span={12}>
            <Search
              placeholder="搜索产品名称或代码"
              allowClear
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              style={{ width: '100%', fontSize: '16px' }}
              size="large"
              prefix={<SearchOutlined style={{ fontSize: '16px' }} />}
            />
          </Col>
          <Col span={12} style={{ textAlign: 'right' }}>
            <Space size="large">
              <Button
                type="primary"
                icon={<DownloadOutlined style={{ fontSize: '16px' }} />}
                loading={exporting}
                onClick={() => handleExport('xlsx')}
                size="large"
                style={{ fontSize: '16px', height: '48px' }}
              >
                导出 Excel
              </Button>
              <Button
                icon={<DownloadOutlined style={{ fontSize: '16px' }} />}
                loading={exporting}
                onClick={() => handleExport('csv')}
                size="large"
                style={{ fontSize: '16px', height: '48px' }}
              >
                导出 CSV
              </Button>
            </Space>
          </Col>
        </Row>

        <Table
          columns={getColumns()}
          dataSource={filteredData}
          rowKey="product_code"
          scroll={{ x: 2400 }}
          size="large"
          style={{ fontSize: '16px' }}
          pagination={{
            total: filteredData.length,
            pageSize: 15,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => 
              <span style={{ fontSize: '16px' }}>第 {range[0]}-{range[1]} 条，共 {total} 条记录</span>,
            style: { fontSize: '16px' }
          }}
        />
      </Card>
    </div>
  );
};

export default AllocationPage;