import axios from 'axios';

// 创建axios实例
const api = axios.create({
  baseURL: '/api',
  timeout: 300000, // 5分钟超时，因为计算可能需要较长时间
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    console.error('请求错误:', error);
    return Promise.reject(error);
  }
);

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url);
    return response;
  },
  (error) => {
    console.error('响应错误:', error);
    if (error.response) {
      // 服务器返回错误状态码
      const { status, data } = error.response;
      console.error(`HTTP ${status}:`, data);
    } else if (error.request) {
      // 请求发送但没有收到响应
      console.error('网络错误:', error.request);
    } else {
      // 其他错误
      console.error('请求配置错误:', error.message);
    }
    return Promise.reject(error);
  }
);

// API方法
export const uploadFile = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  return api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

export const solveAllocation = (config) => {
  return api.post('/solve', config);
};

export const getResult = () => {
  return api.get('/result');
};

export const getConstraints = () => {
  return api.get('/constraints');
};

export const exportResult = (format = 'xlsx') => {
  return api.get('/export', {
    params: { format },
    responseType: 'blob',
  });
};

export default api;