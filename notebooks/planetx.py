import asyncio
import os

import planet

async def main():
    async with planet.Session() as sess:
        client = sess.client('orders')

        requests = create_requests()

        await asyncio.gather(*[
            create_and_download(client, request, DOWNLOAD_DIR)
            for request in requests
        ])


if __name__ == '__main__':
    asyncio.run(main())
def create_request():
    # This is your area of interest for the Orders clipping tool
    oregon_aoi = {
       "type":
       "Polygon",
       "coordinates": [[[-117.558734, 45.229745], [-117.452447, 45.229745],
                        [-117.452447, 45.301865], [-117.558734, 45.301865],
                        [-117.558734, 45.229745]]]
   }

   # In practice, you will use a Data API search to find items, but
   # for this example take them as given.
   oregon_items = ['20200909_182525_1014', '20200909_182524_1014']

   oregon_order = planet.order_request.build_request(
       name='oregon_order',
       products=[
           planet.order_request.product(item_ids=oregon_items,
                                        product_bundle='analytic_udm2',
                                        item_type='PSScene')
       ],
       tools=[planet.order_request.clip_tool(aoi=oregon_aoi)])

   return oregon_order
