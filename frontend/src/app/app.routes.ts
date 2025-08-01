import { Routes } from '@angular/router';
import { HomeComponent } from './pages/home/home';
import { TopStocksComponent } from './pages/top-stocks/top-stocks';
import { StockSearchComponent } from './pages/stock-search/stock-search';

export const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'top-stocks', component: TopStocksComponent },
  { path: 'search', component: StockSearchComponent },
  { path: '**', redirectTo: '' }
];
