import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Dashboard Component
const Dashboard = () => {
  const [health, setHealth] = useState(null);
  const [config, setConfig] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [syncJobs, setSyncJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [connectionTest, setConnectionTest] = useState(null);

  // Fetch system health
  const fetchHealth = async () => {
    try {
      const response = await axios.get(`${API}/health`);
      setHealth(response.data);
    } catch (error) {
      console.error('Health check failed:', error);
      setHealth({ status: 'error', error: error.message });
    }
  };

  // Fetch configuration
  const fetchConfig = async () => {
    try {
      const response = await axios.get(`${API}/sync/config`);
      setConfig(response.data);
    } catch (error) {
      console.error('Failed to fetch config:', error);
    }
  };

  // Fetch conversations
  const fetchConversations = async () => {
    try {
      const response = await axios.get(`${API}/conversations/processed?limit=10`);
      setConversations(response.data);
    } catch (error) {
      console.error('Failed to fetch conversations:', error);
    }
  };

  // Fetch sync jobs
  const fetchSyncJobs = async () => {
    try {
      const response = await axios.get(`${API}/sync/jobs?limit=10`);
      setSyncJobs(response.data);
    } catch (error) {
      console.error('Failed to fetch sync jobs:', error);
    }
  };

  // Test connection
  const testConnection = async () => {
    try {
      setConnectionTest({ status: 'testing' });
      const response = await axios.post(`${API}/test/connection`);
      setConnectionTest(response.data);
    } catch (error) {
      console.error('Connection test failed:', error);
      setConnectionTest({ status: 'error', error: error.message });
    }
  };

  // Manual sync
  const triggerManualSync = async () => {
    try {
      const response = await axios.post(`${API}/sync/manual`);
      alert(`Sync started! Job ID: ${response.data.job_id}`);
      fetchSyncJobs(); // Refresh jobs
    } catch (error) {
      console.error('Manual sync failed:', error);
      alert('Manual sync failed: ' + error.message);
    }
  };

  // Update configuration
  const updateConfig = async (updates) => {
    try {
      await axios.put(`${API}/sync/config`, updates);
      fetchConfig(); // Refresh config
      alert('Configuration updated successfully!');
    } catch (error) {
      console.error('Config update failed:', error);
      alert('Configuration update failed: ' + error.message);
    }
  };

  // Initial data load
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        fetchHealth(),
        fetchConfig(),
        fetchConversations(),
        fetchSyncJobs()
      ]);
      setLoading(false);
    };
    loadData();

    // Set up auto-refresh for health status
    const interval = setInterval(fetchHealth, 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-bold">🧠 ChatGPT ↔ Obsidian Sync</h1>
            </div>
            <div className="flex items-center space-x-4">
              <StatusIndicator health={health} />
              <button
                onClick={testConnection}
                className="bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm"
                disabled={connectionTest?.status === 'testing'}
              >
                {connectionTest?.status === 'testing' ? 'Testing...' : 'Test Connection'}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {['dashboard', 'import', 'conversations', 'settings'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-4 px-1 border-b-2 font-medium text-sm capitalize ${
                  activeTab === tab
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-300 hover:text-white hover:border-gray-300'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'dashboard' && (
          <DashboardTab 
            health={health} 
            config={config} 
            syncJobs={syncJobs}
            connectionTest={connectionTest}
            onManualSync={triggerManualSync}
          />
        )}
        {activeTab === 'import' && <ImportTab />}
        {activeTab === 'conversations' && <ConversationsTab conversations={conversations} />}
        {activeTab === 'settings' && <SettingsTab config={config} onUpdate={updateConfig} />}
      </main>
    </div>
  );
};

// Status Indicator Component
const StatusIndicator = ({ health }) => {
  if (!health) return <div className="w-3 h-3 bg-gray-500 rounded-full"></div>;
  
  const statusColors = {
    healthy: 'bg-green-500',
    degraded: 'bg-yellow-500',
    unhealthy: 'bg-red-500',
    error: 'bg-red-600'
  };
  
  return (
    <div className="flex items-center space-x-2">
      <div className={`w-3 h-3 rounded-full ${statusColors[health.status] || 'bg-gray-500'}`}></div>
      <span className="text-sm capitalize">{health.status}</span>
    </div>
  );
};

// Dashboard Tab
const DashboardTab = ({ health, config, syncJobs, connectionTest, onManualSync }) => {
  return (
    <div className="space-y-6">
      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatusCard title="Database" status={health?.database || 'unknown'} />
        <StatusCard title="LLM Service" status={health?.llm_service || 'unknown'} />
        <StatusCard title="Obsidian Vault" status={health?.obsidian_vault || 'unknown'} />
      </div>

      {/* Connection Test Results */}
      {connectionTest && (
        <div className="bg-gray-800 p-6 rounded-lg">
          <h3 className="text-lg font-semibold mb-4">🔌 Connection Test Results</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-400">Provider</div>
              <div className="font-mono">{connectionTest.provider || 'N/A'}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Model</div>
              <div className="font-mono">{connectionTest.model || 'N/A'}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Status</div>
              <div className={`font-semibold ${
                connectionTest.status === 'connected' ? 'text-green-400' : 
                connectionTest.status === 'testing' ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {connectionTest.status}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Response Time</div>
              <div className="font-mono">
                {connectionTest.response_time_seconds ? `${connectionTest.response_time_seconds}s` : 'N/A'}
              </div>
            </div>
          </div>
          {connectionTest.error && (
            <div className="mt-4 p-3 bg-red-900 border border-red-700 rounded">
              <div className="text-red-300">{connectionTest.error}</div>
            </div>
          )}
        </div>
      )}

      {/* Quick Actions */}
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-4">⚡ Quick Actions</h3>
        <div className="flex space-x-4">
          <button
            onClick={onManualSync}
            className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded font-medium"
          >
            🔄 Manual Sync
          </button>
          <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-medium">
            📥 Import Conversation
          </button>
        </div>
      </div>

      {/* Recent Sync Jobs */}
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-4">📋 Recent Sync Jobs</h3>
        {syncJobs.length > 0 ? (
          <div className="space-y-3">
            {syncJobs.slice(0, 5).map((job) => (
              <div key={job.id} className="flex justify-between items-center p-3 bg-gray-700 rounded">
                <div>
                  <div className="font-medium">{job.job_type}</div>
                  <div className="text-sm text-gray-400">
                    {new Date(job.created_at).toLocaleString()}
                  </div>
                </div>
                <div className="text-right">
                  <div className={`font-semibold ${
                    job.status === 'completed' ? 'text-green-400' :
                    job.status === 'running' ? 'text-yellow-400' :
                    job.status === 'failed' ? 'text-red-400' : 'text-gray-400'
                  }`}>
                    {job.status}
                  </div>
                  <div className="text-sm text-gray-400">
                    {job.items_processed || 0} items
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-gray-400">No sync jobs yet</div>
        )}
      </div>
    </div>
  );
};

// Status Card Component
const StatusCard = ({ title, status }) => {
  const statusColors = {
    healthy: 'text-green-400 border-green-400',
    unhealthy: 'text-red-400 border-red-400',
    vault_missing: 'text-yellow-400 border-yellow-400',
    unknown: 'text-gray-400 border-gray-400'
  };

  return (
    <div className={`bg-gray-800 p-4 rounded-lg border-l-4 ${statusColors[status] || statusColors.unknown}`}>
      <h3 className="font-semibold">{title}</h3>
      <p className="text-sm capitalize mt-1">{status.replace('_', ' ')}</p>
    </div>
  );
};

// Import Tab
const ImportTab = () => {
  const [importData, setImportData] = useState({ content: '', title: '', tags: '' });
  const [importing, setImporting] = useState(false);

  const handleImport = async (e) => {
    e.preventDefault();
    if (!importData.content.trim()) {
      alert('Please enter conversation content');
      return;
    }

    setImporting(true);
    try {
      const response = await axios.post(`${API}/import/chatgpt`, {
        content: importData.content,
        import_type: 'text',
        title: importData.title || 'Imported Conversation',
        tags: importData.tags.split(',').map(t => t.trim()).filter(Boolean)
      });
      
      alert('Conversation imported successfully! Processing in background...');
      setImportData({ content: '', title: '', tags: '' });
    } catch (error) {
      console.error('Import failed:', error);
      alert('Import failed: ' + error.message);
    } finally {
      setImporting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-4">📥 Import ChatGPT Conversation</h3>
        <form onSubmit={handleImport} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Title (optional)</label>
            <input
              type="text"
              value={importData.title}
              onChange={(e) => setImportData({...importData, title: e.target.value})}
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded focus:ring-2 focus:ring-blue-500"
              placeholder="Conversation title..."
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Tags (comma-separated)</label>
            <input
              type="text"
              value={importData.tags}
              onChange={(e) => setImportData({...importData, tags: e.target.value})}
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded focus:ring-2 focus:ring-blue-500"
              placeholder="chatgpt, important, work..."
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Conversation Content</label>
            <textarea
              value={importData.content}
              onChange={(e) => setImportData({...importData, content: e.target.value})}
              rows={12}
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded focus:ring-2 focus:ring-blue-500"
              placeholder="Paste your ChatGPT conversation here..."
            />
          </div>
          <button
            type="submit"
            disabled={importing}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 py-3 rounded font-medium"
          >
            {importing ? 'Importing...' : 'Import & Process'}
          </button>
        </form>
      </div>
    </div>
  );
};

// Conversations Tab
const ConversationsTab = ({ conversations }) => {
  return (
    <div className="space-y-6">
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-4">💬 Processed Conversations</h3>
        {conversations.length > 0 ? (
          <div className="space-y-4">
            {conversations.map((conv) => (
              <div key={conv.id} className="p-4 bg-gray-700 rounded-lg">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-semibold">{conv.structured_title}</h4>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded text-xs ${
                      conv.synced_to_obsidian ? 'bg-green-600' : 'bg-yellow-600'
                    }`}>
                      {conv.synced_to_obsidian ? 'Synced' : 'Pending'}
                    </span>
                  </div>
                </div>
                <p className="text-gray-300 text-sm mb-3">{conv.summary}</p>
                <div className="flex flex-wrap gap-2">
                  {conv.tags.map((tag) => (
                    <span key={tag} className="px-2 py-1 bg-blue-600 rounded text-xs">
                      #{tag}
                    </span>
                  ))}
                </div>
                <div className="mt-3 text-xs text-gray-400">
                  Created: {new Date(conv.created_at).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-gray-400">No conversations processed yet</div>
        )}
      </div>
    </div>
  );
};

// Settings Tab
const SettingsTab = ({ config, onUpdate }) => {
  const [settings, setSettings] = useState(config || {});

  useEffect(() => {
    setSettings(config || {});
  }, [config]);

  const handleSave = () => {
    onUpdate(settings);
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-4">⚙️ Sync Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium mb-2">LLM Provider</label>
            <select
              value={settings.llm_provider || 'openai'}
              onChange={(e) => setSettings({...settings, llm_provider: e.target.value})}
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded focus:ring-2 focus:ring-blue-500"
            >
              <option value="openai">OpenAI</option>
              <option value="ollama">Ollama</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">
              {settings.llm_provider === 'ollama' ? 'Ollama Model' : 'OpenAI Model'}
            </label>
            <input
              type="text"
              value={settings.llm_provider === 'ollama' ? settings.ollama_model : settings.openai_model}
              onChange={(e) => setSettings({
                ...settings,
                [settings.llm_provider === 'ollama' ? 'ollama_model' : 'openai_model']: e.target.value
              })}
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded focus:ring-2 focus:ring-blue-500"
              placeholder={settings.llm_provider === 'ollama' ? 'llama2' : 'gpt-4'}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Sync Interval (minutes)</label>
            <input
              type="number"
              value={settings.sync_interval_minutes || 60}
              onChange={(e) => setSettings({...settings, sync_interval_minutes: parseInt(e.target.value)})}
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Processing Template</label>
            <select
              value={settings.processing_template || 'advanced_structured'}
              onChange={(e) => setSettings({...settings, processing_template: e.target.value})}
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded focus:ring-2 focus:ring-blue-500"
            >
              <option value="basic">Basic</option>
              <option value="advanced_structured">Advanced Structured</option>
            </select>
          </div>
        </div>
        <div className="mt-6 flex items-center">
          <input
            type="checkbox"
            id="auto_sync"
            checked={settings.auto_sync_enabled !== false}
            onChange={(e) => setSettings({...settings, auto_sync_enabled: e.target.checked})}
            className="mr-2"
          />
          <label htmlFor="auto_sync" className="text-sm">Enable automatic sync</label>
        </div>
        <button
          onClick={handleSave}
          className="mt-6 bg-green-600 hover:bg-green-700 px-6 py-2 rounded font-medium"
        >
          Save Configuration
        </button>
      </div>
    </div>
  );
};

// Main App Component
function App() {
  return (
    <div className="App">
      <Dashboard />
    </div>
  );
}

export default App;