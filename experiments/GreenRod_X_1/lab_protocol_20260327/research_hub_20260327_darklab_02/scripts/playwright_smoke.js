const { chromium } = require('playwright');

const baseUrl = process.env.HUB_URL || 'http://127.0.0.1:8000/';

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1680, height: 1020 } });

  await page.goto(baseUrl, { waitUntil: 'networkidle' });
  await page.waitForSelector('#sotaCards .sota-card', { timeout: 20000 });
  await page.waitForSelector('#ablationCards .ablation-card', { timeout: 20000 });
  await page.waitForSelector('#statusChart canvas', { timeout: 20000 });
  await page.waitForSelector('#timelineChart canvas', { timeout: 20000 });

  await page.screenshot({ path: 'artifacts/playwright/01_overview.png', fullPage: true });

  await page.fill('#searchInput', 'proxy');
  await page.selectOption('#statusFilter', 'warn');
  await page.waitForTimeout(900);

  const firstRow = page.locator('#recordsBody .record-row').first();
  if (await firstRow.count()) {
    await firstRow.click();
  }

  await page.waitForTimeout(700);
  await page.screenshot({ path: 'artifacts/playwright/02_filtered_detail.png', fullPage: true });

  await browser.close();
})();
