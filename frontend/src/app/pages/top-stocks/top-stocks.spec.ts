import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TopStocks } from './top-stocks';

describe('TopStocks', () => {
  let component: TopStocks;
  let fixture: ComponentFixture<TopStocks>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TopStocks]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TopStocks);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
