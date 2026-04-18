document.addEventListener('DOMContentLoaded', () => {
    fetchData();
});

async function fetchData() {
    try {
        const response = await fetch('data.json');
        const data = await response.json();
        
        updateHeader(data);
        renderLeaderboard(data.leaderboard);
        renderLogs(data.latest_logs);
        updateStatus(data);
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

function updateHeader(data) {
    document.getElementById('project-name').textContent = data.project_name || 'Parameter Golf';
    document.getElementById('project-desc').textContent = data.description || '';
}

function renderLeaderboard(leaderboard) {
    const tbody = document.getElementById('leaderboard-body');
    tbody.innerHTML = '';
    
    leaderboard.forEach((entry, index) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="rank">#${index + 1}</td>
            <td>${entry.Run}</td>
            <td class="score">${entry.Score}</td>
            <td>${entry.Author}</td>
            <td>${entry.Date}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderLogs(logs) {
    const logsList = document.getElementById('logs-list');
    logsList.innerHTML = '';
    
    if (!logs || logs.length === 0) {
        logsList.innerHTML = '<p class="log-item">No recent diagnostics found.</p>';
        return;
    }
    
    logs.forEach(log => {
        const div = document.createElement('div');
        div.className = 'log-item';
        const date = new Date(log.timestamp);
        div.innerHTML = `
            <div class="log-name">${log.name}</div>
            <div class="log-time">${date.toLocaleString()}</div>
        `;
        logsList.appendChild(div);
    });
}

function updateStatus(data) {
    const lastUpdated = document.getElementById('last-updated');
    const date = new Date(data.last_updated);
    lastUpdated.textContent = date.toLocaleTimeString() + ' ' + date.toLocaleDateString();
}
