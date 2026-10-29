import lumeapp from 'liblumeapp.so';

@Entry
@Component
struct Index {
  private xComponentId: string = 'lumeapp_xcomponent';
  private intervalId: number = -1;
  @State activeModel: string = 'doll';

  aboutToAppear(): void {
    let context = getContext(this) as Context;
    let resMgr = context.resourceManager;
    let filesDir = context.filesDir;
    lumeapp.initResourceManager(resMgr, filesDir);
    console.info('[LumeApp] ResourceManager + filesDir passed to native: ' + filesDir);
  }

  private selectModel(name: string): void {
    if (this.activeModel === name) {
      return;
    }
    this.activeModel = name;
    lumeapp.setModel(name);
    console.info('[LumeApp] switch model to ' + name);
  }

  build() {
    Stack({ alignContent: Alignment.Bottom }) {
      XComponent({
        id: this.xComponentId,
        type: XComponentType.SURFACE,
        libraryname: 'lumeapp'
      })
        .onLoad(() => {
          console.info('[LumeApp] XComponent onLoad - starting render loop');
          this.intervalId = setInterval(() => {
            lumeapp.drawFrame();
          }, 16);
        })
        .onDestroy(() => {
          console.info('[LumeApp] XComponent onDestroy');
          if (this.intervalId >= 0) {
            clearInterval(this.intervalId);
            this.intervalId = -1;
          }
        })
        .width('100%')
        .height('100%')

      Row({ space: 10 }) {
        Button('Fruit bowl')
          .fontSize(13)
          .height(34)
          .backgroundColor(this.activeModel === 'fruit_bowl' ? '#FF6A00' : '#66000000')
          .onClick(() => this.selectModel('fruit_bowl'))
        Button('Godzilla')
          .fontSize(13)
          .height(34)
          .backgroundColor(this.activeModel === 'godzilla' ? '#FF6A00' : '#66000000')
          .onClick(() => this.selectModel('godzilla'))
        Button('Doll')
          .fontSize(13)
          .height(34)
          .backgroundColor(this.activeModel === 'doll' ? '#FF6A00' : '#66000000')
          .onClick(() => this.selectModel('doll'))
      }
      .margin({ bottom: 32 })
    }
    .width('100%')
    .height('100%')
  }
}
