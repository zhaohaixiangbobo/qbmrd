


import React, { useState, useEffect } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { Layout, Menu, Typography, theme } from 'antd';
import { 
  HomeOutlined, 
  TableOutlined, 
  BarChartOutlined,
  UploadOutlined 
} from '@ant-design/icons';

import HomePage from './pages/HomePage';
import AllocationPage from './pages/AllocationPage';
import ResultPage from './pages/ResultPage';
import StatisticsPage from './pages/StatisticsPage';

const { Header, Content, Sider } = Layout;
const { Title } = Typography;

const App = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [selectedKey, setSelectedKey] = useState('1');
  const navigate = useNavigate();
  const location = useLocation();
  
  const {
    token: { colorBgContainer },
  } = theme.useToken();

  const menuItems = [
    {
      key: '1',
      icon: <HomeOutlined />,
      label: '首页 - 数据导入与配置',
      path: '/'
    },
    {
      key: '2',
      icon: <TableOutlined />,
      label: '分配明细查看 & 导出',
      path: '/allocation'
    },
    {
      key: '3',
      icon: <BarChartOutlined />,
      label: '结果统计分析',
      path: '/statistics'
    },
    {
      key: '4',
      icon: <BarChartOutlined />,
      label: '约束验证页面',
      path: '/result'
    },
    {
      key: '5',
      icon: <BarChartOutlined />,
      label: '数据推送',
      path: '/statistics'
    }
  ];

  // 根据当前路径设置选中的菜单项
  useEffect(() => {
    const currentItem = menuItems.find(item => item.path === location.pathname);
    if (currentItem) {
      setSelectedKey(currentItem.key);
    }
  }, [location.pathname]);

  const handleMenuClick = ({ key }) => {
    const selectedItem = menuItems.find(item => item.key === key);
    if (selectedItem) {
      setSelectedKey(key);
      navigate(selectedItem.path);
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider 
        collapsible 
        collapsed={collapsed} 
        onCollapse={setCollapsed}
        width={320}
        style={{
          position: 'fixed',
          height: '100vh',
          left: 0,
          top: 0,
          zIndex: 1000
        }}
      >
        <div style={{ 
          height: 56, 
          margin: 16, 
          background: 'rgba(255, 255, 255, 0.2)',
          borderRadius: 10,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          fontWeight: 'bold',
          fontSize: '20px'
        }}>
          {collapsed ? '货源' : '卷烟货源分配平台'}
        </div>
        <Menu
          theme="dark"
          selectedKeys={[selectedKey]}
          mode="inline"
          onClick={handleMenuClick}
          style={{
            fontSize: '18px',
            lineHeight: '1.6'
          }}
          items={menuItems.map(item => ({
            key: item.key,
            icon: React.cloneElement(item.icon, { style: { fontSize: '20px' } }),
            label: item.label,
            style: {
              fontSize: '18px',
              height: '60px',
              lineHeight: '60px',
              marginBottom: '8px',
              borderRadius: '8px',
              margin: '4px 8px'
            }
          }))}
        />
      </Sider>
      <Layout style={{ marginLeft: collapsed ? 80 : 320, transition: 'margin-left 0.2s' }}>
        <Header style={{ 
          padding: '0 40px',
          background: colorBgContainer,
          display: 'flex',
          alignItems: 'center',
          height: '88px',
          position: 'sticky',
          top: 0,
          zIndex: 999,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
        }}>
          <Title level={2} style={{ margin: 0, color: '#1890ff', fontSize: '32px' }}>
            <UploadOutlined style={{ marginRight: 16, fontSize: '36px' }} />
            卷烟货源分配优化系统
          </Title>
        </Header>
        <Content style={{ 
          margin: '32px 24px', 
          padding: 32, 
          minHeight: 280, 
          background: colorBgContainer,
          borderRadius: 10
        }}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/allocation" element={<AllocationPage />} />
            <Route path="/statistics" element={<StatisticsPage />} />
            <Route path="/result" element={<ResultPage />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;