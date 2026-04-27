import { expect, test, type Page } from '@playwright/test';

async function forceStaticFallback(page: Page) {
  await page.route('http://localhost:8080/**', (route) => route.abort());
  await page.route('http://127.0.0.1:8080/**', (route) => route.abort());
  await page.route('**/wasm/**', (route) => route.abort());
}

async function saveScreenshot(page: Page, name: string) {
  await page.screenshot({
    path: test.info().outputPath(`${name}.png`),
    fullPage: false,
  });
}

test.describe('Fast MNIST interactive demo', () => {
  test.beforeEach(async ({ page }) => {
    await forceStaticFallback(page);
  });

  test('renders the first viewport without hiding the actual tool', async ({ page }) => {
    await page.goto('/index.html');

    await expect(page.getByRole('heading', { name: /Fast MNIST Neural Network/i })).toBeVisible();
    await expect(page.getByRole('heading', { name: /Draw Here/i })).toBeVisible();
    await expect(page.getByRole('heading', { name: /Prediction/i })).toBeVisible();
    await expect(page.getByText(/browser fallback ready/i)).toBeVisible();

    const canvasBox = await page.locator('.drawing-canvas').boundingBox();
    const viewport = page.viewportSize();
    expect(canvasBox, 'drawing canvas should be present').not.toBeNull();
    expect(canvasBox!.y, 'drawing canvas should start inside the first viewport').toBeLessThan(
      viewport?.height ?? 1080,
    );

    await saveScreenshot(page, 'first-viewport');
  });

  test('opens command palette and runs the sample digit fallback path', async ({ page }) => {
    await page.goto('/index.html');
    await page.keyboard.press(process.platform === 'darwin' ? 'Meta+K' : 'Control+K');

    const dialog = page.getByRole('dialog', { name: /command palette/i });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByText(/Load sample digit/i)).toBeVisible();
    await page.waitForTimeout(250);
    await saveScreenshot(page, 'command-palette');

    await dialog.getByText(/Load sample digit/i).click();

    await expect(page.locator('.predicted-digit')).toBeVisible();
    await expect(page.locator('.runtime-badge')).toHaveText(/js-demo-mode|wasm-mode/);
    await expect(page.locator('.confidence-chart')).toBeVisible();
    await expect(page.locator('.activation-panels')).toBeVisible();

    await saveScreenshot(page, 'sample-prediction');
  });

  test('keeps command palette and drawing workflow usable on mobile', async ({
    page,
    isMobile,
  }) => {
    test.skip(!isMobile, 'mobile-only audit');

    await page.goto('/index.html');
    await expect(page.getByRole('heading', { name: /Fast MNIST Neural Network/i })).toBeVisible();

    await page.keyboard.press(process.platform === 'darwin' ? 'Meta+K' : 'Control+K');
    const dialog = page.getByRole('dialog', { name: /command palette/i });
    await expect(dialog).toBeVisible();
    await dialog.getByText(/Go to drawing canvas/i).click();
    await expect(page.getByRole('heading', { name: /Draw Here/i })).toBeVisible();

    await page.keyboard.press(process.platform === 'darwin' ? 'Meta+K' : 'Control+K');
    await page
      .getByRole('dialog', { name: /command palette/i })
      .getByText(/Load sample digit/i)
      .click();
    await expect(page.locator('.predicted-digit')).toBeVisible();

    await saveScreenshot(page, 'mobile-sample-prediction');
  });

  test('shows the Motion pipeline cards during scroll', async ({ page }) => {
    await page.goto('/index.html');
    await page.evaluate(() => {
      document.getElementById('pipeline')?.scrollIntoView({
        behavior: 'auto',
        block: 'center',
      });
    });
    await page.waitForTimeout(600);

    await expect(page.getByText('You draw.')).toBeVisible();
    await expect(page.getByText('C++ classifies.')).toBeVisible();
    await expect(page.getByText('You see the answer.')).toBeVisible();

    await saveScreenshot(page, 'pipeline-section');
  });
});
